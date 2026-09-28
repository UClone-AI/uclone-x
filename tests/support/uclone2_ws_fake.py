"""A loopback uClone2 socket endpoint for the link session tests.

`FakeLinkedSocket` serves `GET /api/v4/linked/ws` on 127.0.0.1 with an ephemeral port, so
the unit network guard admits the connect (P8) and nothing leaves the machine. It plays the
server's side of the frames contract: it refuses the upgrade with an HTTP status when told
to, sends `hello` on upgrade, pings when told to, records every frame the runtime sends,
and closes or drops a socket on command.

It never closes a socket on its own. A fake that ended the conversation by itself would end
the session's loop in a way the real server never does, and a reconnect test would pass
for the wrong reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Self, cast

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from tests.support.uclone2_fake import BOT_ID

JSON = dict[str, object]

WS_PATH: Final = "/api/v4/linked/ws"

HELLO_LIMITS: Final[JSON] = {
    "guestbook_reply_max_chars": 300,
    "post_title_max_chars": 80,
    "comment_max_chars": 500,
}
HELLO_UPDATED_AT: Final = "2026-09-26T22:40:00Z"


def hello_frame(
    *, limits: JSON | None = None, clone_updated_at: str = HELLO_UPDATED_AT, pending: int = 3
) -> JSON:
    return {
        "type": "hello",
        "protocol": 1,
        "bot_id": BOT_ID,
        "server_time": "2026-09-27T09:00:00Z",
        "pending": pending,
        "limits": dict(limits if limits is not None else HELLO_LIMITS),
        "clone_updated_at": clone_updated_at,
    }


def _quiet_logger() -> logging.Logger:
    """The fake's own logger, outside the logging tree.

    The server side logs each request header at DEBUG, the runtime's `Authorization`
    included. The token tests capture every log record the runtime emits; the fake's must
    not be among them, or a leak in the fake would read as a leak in the runtime.
    """
    quiet = logging.Logger("uclone2_ws_fake", level=logging.CRITICAL)
    quiet.propagate = False
    quiet.addHandler(logging.NullHandler())
    return quiet


@dataclass
class FakeConnection:
    """One socket the runtime opened: what it sent, and how it ended."""

    ws: ServerConnection
    path: str
    headers: Headers
    received: list[JSON | str] = field(default_factory=list[JSON | str])
    #: The close code the runtime sent, when it closed the socket itself.
    client_close_code: int | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    def frames(self, frame_type: str | None = None) -> list[JSON]:
        """The JSON frames received, optionally only those of one `type`."""
        out = [f for f in self.received if isinstance(f, dict)]
        return [f for f in out if frame_type is None or f.get("type") == frame_type]

    async def send(self, frame: JSON | str) -> None:
        await self.ws.send(frame if isinstance(frame, str) else json.dumps(frame))

    async def close_with(self, code: int, bye_reason: str | None = None) -> None:
        """What the server does to end a socket: `bye{reason}`, then the matching close."""
        with contextlib.suppress(ConnectionClosed):
            if bye_reason is not None:
                await self.send({"type": "bye", "reason": bye_reason})
            await self.ws.close(code, bye_reason or "")

    def drop(self) -> None:
        """A network drop: the TCP connection goes away with no close frame."""
        self.ws.transport.abort()


class FakeLinkedSocket:
    """`async with FakeLinkedSocket() as server:`; `server.ws_url` is the link's `ws_url`."""

    def __init__(
        self,
        *,
        hello: JSON | None = None,
        refuse_status: int | None = None,
        ping_every_s: float | None = None,
    ) -> None:
        self.hello: JSON | None = hello if hello is not None else hello_frame()
        #: Answer the upgrade with this HTTP status instead of switching protocols.
        self.refuse_status = refuse_status
        self.ping_every_s = ping_every_s
        self.connections: list[FakeConnection] = []
        #: Every upgrade request, including refused ones: (path, headers).
        self.requests: list[tuple[str, Headers]] = []
        self._server: Server | None = None
        self._port = 0

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self._port}{WS_PATH}"

    @property
    def latest(self) -> FakeConnection:
        return self.connections[-1]

    async def __aenter__(self) -> Self:
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            process_request=self._process_request,
            ping_interval=None,
            logger=_quiet_logger(),
        )
        sock = next(iter(self._server.sockets))
        self._port = cast(tuple[str, int], sock.getsockname())[1]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._server.close(close_connections=False)
            for conn in self.connections:
                conn.ws.transport.abort()
            await self._server.wait_closed()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        self.requests.append((request.path, request.headers))
        if self.refuse_status is not None:
            return connection.respond(self.refuse_status, "refused\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        request = ws.request
        assert request is not None
        conn = FakeConnection(ws=ws, path=request.path, headers=request.headers)
        self.connections.append(conn)
        pinger: asyncio.Task[None] | None = None
        try:
            if self.hello is not None:
                await conn.send(self.hello)
            if self.ping_every_s is not None:
                pinger = asyncio.create_task(self._ping(conn, self.ping_every_s))
            async for message in ws:
                text = message if isinstance(message, str) else message.decode()
                try:
                    conn.received.append(cast(JSON, json.loads(text)))
                except ValueError:
                    conn.received.append(text)
        except ConnectionClosed:
            pass
        finally:
            if pinger is not None:
                pinger.cancel()
            close = ws.protocol.close_rcvd
            conn.client_close_code = close.code if close is not None else None
            conn.closed.set()

    @staticmethod
    async def _ping(conn: FakeConnection, every_s: float) -> None:
        with contextlib.suppress(ConnectionClosed):
            while True:
                await asyncio.sleep(every_s)
                await conn.send({"type": "ping", "ts": "2026-09-27T09:00:20Z"})


async def eventually(
    check: Callable[[], bool] | Callable[[], Awaitable[bool]],
    *,
    timeout_s: float = 5.0,
    what: str = "the condition",
) -> None:
    """Poll `check` until it holds; fail with `what` if it does not within `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = check()
        if isinstance(result, Awaitable):
            result = await result
        if result:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)
