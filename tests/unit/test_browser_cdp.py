"""The CDP client against a scripted WebSocket server on loopback (design §3.10)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from uclone_x.browser.cdp import CdpClosedError, CdpConnection, CdpError

Handler = Callable[[ServerConnection, dict[str, Any]], Awaitable[None]]


async def _endpoint(handler: Handler) -> AsyncIterator[str]:
    async def _serve(ws: ServerConnection) -> None:
        async for raw in ws:
            await handler(ws, json.loads(raw))

    async with serve(_serve, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


async def _chrome_like(ws: ServerConnection, message: dict[str, Any]) -> None:
    method = message["method"]
    if method == "Runtime.evaluate":
        # An event for the session first, then the answer, as Chrome interleaves them.
        await ws.send(
            json.dumps(
                {
                    "method": "Page.loadEventFired",
                    "params": {"timestamp": 1},
                    "sessionId": message.get("sessionId"),
                }
            )
        )
        await ws.send(json.dumps({"id": message["id"], "result": {"value": 2}}))
    elif method == "Page.navigate":
        await ws.send(
            json.dumps({"id": message["id"], "error": {"code": -32000, "message": "bad url"}})
        )
    elif method == "Browser.close":
        await ws.close()
    # Anything else goes unanswered.


@pytest.fixture
async def conn() -> AsyncIterator[CdpConnection]:
    async for url in _endpoint(_chrome_like):
        connection = await CdpConnection.connect(url)
        yield connection
        await connection.close()


async def test_a_command_returns_its_result(conn: CdpConnection) -> None:
    assert await conn.send("Runtime.evaluate", {"expression": "1+1"}) == {"value": 2}


async def test_an_error_answer_raises_with_the_method_named(conn: CdpConnection) -> None:
    with pytest.raises(CdpError) as raised:
        await conn.send("Page.navigate", {"url": "x"}, session_id="S1")

    assert raised.value.method == "Page.navigate"
    assert raised.value.code == -32000
    assert raised.value.message == "bad url"


async def test_events_reach_only_their_session(conn: CdpConnection) -> None:
    mine = conn.subscribe("S1")
    other = conn.subscribe("S2")

    await conn.send("Runtime.evaluate", {}, session_id="S1")

    event = mine.get_nowait()
    assert (event.method, event.session_id) == ("Page.loadEventFired", "S1")
    assert other.empty()

    conn.unsubscribe("S1", mine)
    await conn.send("Runtime.evaluate", {}, session_id="S1")
    assert mine.empty()


async def test_an_unanswered_command_times_out(conn: CdpConnection) -> None:
    with pytest.raises(TimeoutError):
        await conn.send("Page.enable", timeout=0.05)


async def test_a_closed_connection_fails_the_waiting_command(conn: CdpConnection) -> None:
    waiting = asyncio.create_task(conn.send("Page.enable"))
    await asyncio.sleep(0.05)

    with pytest.raises(CdpClosedError):
        await conn.send("Browser.close")
    with pytest.raises(CdpClosedError):
        await waiting

    assert conn.closed
    with pytest.raises(CdpClosedError):
        await conn.send("Runtime.evaluate")
