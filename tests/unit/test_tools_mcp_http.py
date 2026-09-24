"""MCP client transports: Streamable HTTP, and stdio reply matching.

The HTTP server here is an `httpx.MockTransport` handler that speaks the Streamable HTTP
transport's side of the exchange, so nothing leaves the process (P8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from uclone_x.sandbox.models import NoIsolation
from uclone_x.tools.client import MCPClient, exposed_tool_name
from uclone_x.tools.models import MCPConnectionConfig, MCPTransport, ToolContext

URL = "https://mcp.example.test/mcp"


class FakeHttpServer:
    """Records every request and answers as a Streamable HTTP MCP server would."""

    def __init__(self, *, sse: bool = False, status: int = 200) -> None:
        self.sse = sse
        self.status = status
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200)
        body: dict[str, Any] = json.loads(request.content)
        self.bodies.append(body)
        if self.status != 200:
            return httpx.Response(self.status, text="denied")
        if "id" not in body:
            return httpx.Response(202)
        method = body["method"]
        result: dict[str, Any]
        headers: dict[str, str] = {}
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}
            headers["Mcp-Session-Id"] = "sess-42"
        elif method == "tools/list":
            if body["params"].get("cursor") == "page2":
                result = {"tools": [{"name": "fetch.page", "description": "Fetch"}]}
            else:
                result = {
                    "tools": [{"name": "search", "description": "Search", "inputSchema": {}}],
                    "nextCursor": "page2",
                }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": f"called {body['params']['name']}"}]}
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {}})
        reply = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        if self.sse:
            notice: dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {},
            }
            stream = (
                f"event: message\ndata: {json.dumps(notice)}\n\n"
                f"event: message\ndata: {json.dumps(reply)}\n\n"
            )
            headers["content-type"] = "text/event-stream"
            return httpx.Response(200, headers=headers, content=stream.encode())
        return httpx.Response(200, headers=headers, json=reply)


def _client(server: FakeHttpServer, prefix: str | None = "web") -> MCPClient:
    config = MCPConnectionConfig(
        server_name="web",
        transport=MCPTransport.STREAMABLE_HTTP,
        url=URL,
        headers={"Authorization": "Bearer secret-token"},
        allow_network=True,
        isolation=NoIsolation(),
    )
    return MCPClient(
        config=config,
        tool_name_prefix=prefix,
        http_transport=httpx.MockTransport(server.handler),
    )


@pytest.mark.asyncio
async def test_http_session_id_is_sent_after_initialize() -> None:
    """The session the server assigned at initialize is named on every later request.

    Killed by: src/uclone_x/tools/client.py :: self._http_session_id = session_id
    Becomes: pass
    """
    server = FakeHttpServer()
    client = _client(server)
    await client.connect()
    await client.list_tools()
    later = [r for r in server.requests[1:] if r.method == "POST"]
    assert later, "nothing was sent after initialize"
    assert all(r.headers.get("mcp-session-id") == "sess-42" for r in later)
    assert all(r.headers.get("mcp-protocol-version") == "2025-06-18" for r in later)
    await client.disconnect()


@pytest.mark.asyncio
async def test_http_configured_headers_reach_the_server() -> None:
    """A sign-in header the user configured is sent with every request.

    Killed by: src/uclone_x/tools/client.py :: **dict(self._config.headers),
    Becomes: **{},
    """
    server = FakeHttpServer()
    client = _client(server)
    await client.connect()
    assert server.requests[0].headers["authorization"] == "Bearer secret-token"
    accept = server.requests[0].headers["accept"]
    assert "application/json" in accept and "text/event-stream" in accept
    await client.disconnect()


@pytest.mark.asyncio
async def test_http_tools_list_follows_pagination_and_prefixes_names() -> None:
    """Every page is read, and names are namespaced and made provider-safe.

    Killed by: src/uclone_x/tools/client.py :: cursor = next_cursor
    Becomes: break
    """
    server = FakeHttpServer()
    client = _client(server)
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["web__search", "web__fetch_page"]
    await client.disconnect()


@pytest.mark.asyncio
async def test_prefixed_tool_calls_the_server_by_its_own_name(tmp_path: Path) -> None:
    """The model sees `web__fetch_page`; the server is asked for `fetch.page`.

    Killed by: src/uclone_x/tools/client.py :: return await self._client.call_tool(self._remote_name, params, context)
    Becomes: return await self._client.call_tool(self._name, params, context)
    """
    server = FakeHttpServer()
    client = _client(server)
    tools = await client.list_tools()
    fetch = next(t for t in tools if t.name == "web__fetch_page")
    ctx = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)
    result = await fetch.execute({"url": "x"}, ctx)
    assert result.success is True
    assert server.bodies[-1]["params"]["name"] == "fetch.page"
    assert result.output == [{"type": "text", "text": "called fetch.page"}]
    await client.disconnect()


@pytest.mark.asyncio
async def test_http_event_stream_reply_skips_interleaved_notifications(tmp_path: Path) -> None:
    """An SSE answer may carry notifications before the reply; the reply is the one taken.

    Killed by: src/uclone_x/tools/client.py :: if message is not None and message.get("id") == want_id:
    Becomes: if message is not None:
    """
    server = FakeHttpServer(sse=True)
    client = _client(server)
    tools = await client.list_tools()
    assert [t.name for t in tools] == ["web__search", "web__fetch_page"]
    await client.disconnect()


@pytest.mark.asyncio
async def test_http_refusal_names_the_missing_sign_in() -> None:
    """A 401 says the server wants a sign-in header, not just a status code (P6)."""
    server = FakeHttpServer(status=401)
    client = _client(server)
    with pytest.raises(RuntimeError, match="Authorization"):
        await client.connect()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_http_disconnect_ends_the_session() -> None:
    """Disconnecting tells the server the session is over (DELETE with its id).

    Killed by: src/uclone_x/tools/client.py :: await http.delete(self._config.url, headers=self._http_headers(), timeout=2.0)
    Becomes: pass
    """
    server = FakeHttpServer()
    client = _client(server)
    await client.connect()
    await client.disconnect()
    deletes = [r for r in server.requests if r.method == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0].headers["mcp-session-id"] == "sess-42"


def test_exposed_tool_name_is_provider_safe() -> None:
    """Names are `<prefix>__<tool>`, restricted to [A-Za-z0-9_-], and at most 64 long."""
    assert exposed_tool_name(None, "read.file") == "read.file"
    assert exposed_tool_name("gh", "issues/list") == "gh__issues_list"
    assert len(exposed_tool_name("p", "x" * 200)) == 64


_CHATTY_SERVER = r"""
import json, sys
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    if req.get("method") == "initialize":
        send({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "2024-11-05"}})
        continue
    if req.get("method") == "tools/list":
        # A log line, a notification and a ping arrive before the reply.
        sys.stdout.write("server warming up\n"); sys.stdout.flush()
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "hi"}})
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
        pong = json.loads(sys.stdin.readline())
        assert pong == {"jsonrpc": "2.0", "id": "srv-1", "result": {}}, pong
        send({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [{"name": "echo"}]}})
"""


@pytest.mark.asyncio
async def test_stdio_reply_is_matched_past_notifications_and_pings(tmp_path: Path) -> None:
    """A notification or ping on the pipe is not mistaken for the reply.

    Killed by: src/uclone_x/tools/client.py :: if message.get("id") == want_id and ("result" in message or "error" in message):
    Becomes: if True:
    """
    script = tmp_path / "chatty.py"
    script.write_text(_CHATTY_SERVER)
    config = MCPConnectionConfig(
        server_name="chatty",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=(str(script),),
        workspace_root=tmp_path,
    )
    async with MCPClient(config=config, connect_timeout=5.0) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools] == ["echo"]


@pytest.mark.asyncio
async def test_http_redirect_is_not_followed() -> None:
    """A redirect would carry every sign-in header but `Authorization` to the new host.

    Killed by: src/uclone_x/tools/client.py :: if response.is_redirect:
    Becomes: if False:
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://elsewhere.test/mcp"})

    config = MCPConnectionConfig(
        server_name="web",
        transport=MCPTransport.STREAMABLE_HTTP,
        url=URL,
        headers={"X-API-Key": "k-1"},
        allow_network=True,
        isolation=NoIsolation(),
    )
    client = MCPClient(config=config, http_transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="elsewhere.test"):
        await client.connect()
    assert seen == [URL]


@pytest.mark.asyncio
async def test_http_oversize_reply_is_refused_unread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply past the cap is refused by size, not held in memory until a timeout.

    Killed by: src/uclone_x/tools/client.py :: if len(raw) > _MAX_HTTP_MESSAGE:
    Becomes: if False:
    """
    import uclone_x.tools.client as client_module

    monkeypatch.setattr(client_module, "_MAX_HTTP_MESSAGE", 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "pad": "x" * 4096})

    config = MCPConnectionConfig(
        server_name="web",
        transport=MCPTransport.STREAMABLE_HTTP,
        url=URL,
        allow_network=True,
        isolation=NoIsolation(),
    )
    client = MCPClient(config=config, http_transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="larger than"):
        await client.connect()


_NOISY_SERVER = r"""
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    # asyncio buffers an unread stderr itself up to twice the stream limit (32 MiB here),
    # then stops reading; past that the pipe fills and the server blocks on this write.
    sys.stderr.write("x" * (40 * 1024 * 1024)); sys.stderr.flush()
    result = {"protocolVersion": "2024-11-05"} if req["method"] == "initialize" else {"tools": []}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


@pytest.mark.asyncio
async def test_stdio_server_writing_much_to_stderr_does_not_stall(tmp_path: Path) -> None:
    """stderr is drained as it arrives, so a talkative server keeps answering, in bounded memory.

    Killed by: src/uclone_x/tools/client.py :: self._stderr_task = asyncio.create_task(self._drain_stderr(self._process))
    Becomes: self._stderr_task = None
    """
    script = tmp_path / "noisy.py"
    script.write_text(_NOISY_SERVER)
    config = MCPConnectionConfig(
        server_name="noisy",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=(str(script),),
        workspace_root=tmp_path,
    )
    async with MCPClient(config=config, connect_timeout=10.0) as client:
        assert await client.list_tools() == []
        assert len(client._stderr_tail) <= 4096  # pyright: ignore[reportPrivateUsage]
