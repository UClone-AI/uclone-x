"""External MCP servers: the manager that stores and connects them, and its HTTP routes.

Remote servers here are an `httpx.MockTransport` handler; nothing leaves the process (P8).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from uclone_x.tools.client import MCPClient
from uclone_x.tools.mcp_manager import (
    DuplicateServerError,
    MCPServerManager,
    MCPServerSpec,
    spec_from_entry,
)
from uclone_x.tools.models import MCPConnectionConfig
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import create_ui_app


def _mcp_handler(tool_names: tuple[str, ...], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        if status != 200:
            return httpx.Response(status, text="no")
        body: dict[str, Any] = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202)
        if body["method"] == "initialize":
            result: dict[str, Any] = {"protocolVersion": "2025-06-18"}
        elif body["method"] == "tools/list":
            result = {"tools": [{"name": n, "description": f"does {n}"} for n in tool_names]}
        else:
            result = {"content": [{"type": "text", "text": "ok"}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    return httpx.MockTransport(handler)


def _factory(tool_names: tuple[str, ...] = ("search", "fetch"), status: int = 200):  # noqa: ANN202
    def make(config: MCPConnectionConfig, prefix: str) -> MCPClient:
        return MCPClient(
            config=config,
            tool_name_prefix=prefix,
            http_transport=_mcp_handler(tool_names, status),
        )

    return make


def _remote(name: str = "web", token: str = "tok-123") -> MCPServerSpec:
    return MCPServerSpec(
        name=name,
        transport="http",
        url="https://mcp.example.test/mcp",
        headers={"Authorization": f"Bearer {token}"},
    )


def _manager(tmp_path: Path, **kw: Any) -> tuple[MCPServerManager, ToolRegistry]:
    registry = ToolRegistry()
    manager = MCPServerManager(
        registry=registry,
        config_path=tmp_path / "mcp_servers.json",
        workspace_root=tmp_path,
        client_factory=kw.pop("client_factory", _factory()),
        **kw,
    )
    return manager, registry


@pytest.mark.asyncio
async def test_added_server_registers_prefixed_tools(tmp_path: Path) -> None:
    """A connected server's tools reach the registry clones read from, namespaced.

    Killed by: src/uclone_x/tools/mcp_manager.py :: self._registry.register(tool)
    Becomes: pass
    """
    manager, registry = _manager(tmp_path)
    view = await manager.add(_remote())
    assert view["status"] == "connected"
    assert sorted(t.name for t in registry.list_tools()) == ["web__fetch", "web__search"]
    assert [t["name"] for t in view["tools"]] == ["web__search", "web__fetch"]
    await manager.close()


@pytest.mark.asyncio
async def test_removed_server_takes_its_tools_away(tmp_path: Path) -> None:
    """Removing or disabling a server unregisters its tools; nothing stale stays callable.

    Killed by: src/uclone_x/tools/mcp_manager.py :: self._registry.unregister(name)
    Becomes: pass
    """
    manager, registry = _manager(tmp_path)
    await manager.add(_remote())
    await manager.set_enabled("web", False)
    assert registry.list_tools() == []
    assert manager.views()[0]["status"] == "disabled"
    await manager.set_enabled("web", True)
    assert len(registry.list_tools()) == 2
    await manager.remove("web")
    assert registry.list_tools() == []
    assert manager.views() == []


@pytest.mark.asyncio
async def test_colliding_tool_name_is_reported_not_overwritten(tmp_path: Path) -> None:
    """An existing tool with the same name is kept, and the user is told which was skipped.

    Killed by: src/uclone_x/tools/mcp_manager.py :: if self._registry.get(tool.name) is not None:
    Becomes: if False:
    """
    manager, registry = _manager(tmp_path)
    await manager.add(_remote())
    original = registry.get("web__search")
    manager2 = MCPServerManager(
        registry=registry,
        config_path=tmp_path / "other.json",
        workspace_root=tmp_path,
        client_factory=_factory(("search",)),
    )
    view = await manager2.add(_remote())
    assert registry.get("web__search") is original
    assert "web__search" in str(view["error"])
    assert view["tools"] == []
    await manager.close()
    await manager2.close()


@pytest.mark.asyncio
async def test_failed_connection_is_saved_with_its_cause(tmp_path: Path) -> None:
    """A server that refuses is still saved, and says why, so the user can fix and retry."""
    manager, registry = _manager(tmp_path, client_factory=_factory(status=401))
    view = await manager.add(_remote())
    assert view["status"] == "error"
    assert "Authorization" in str(view["error"])
    assert registry.list_tools() == []
    reloaded, _ = _manager(tmp_path)
    assert [v["name"] for v in reloaded.views()] == ["web"]


@pytest.mark.asyncio
async def test_servers_persist_and_reconnect_on_start(tmp_path: Path) -> None:
    """What the user added survives a restart and connects again at startup."""
    manager, _ = _manager(tmp_path)
    await manager.add(_remote())
    await manager.close()
    reloaded, registry = _manager(tmp_path)
    assert reloaded.views()[0]["status"] == "connecting"
    await reloaded.start()
    assert reloaded.views()[0]["status"] == "connected"
    assert len(registry.list_tools()) == 2
    await reloaded.close()


@pytest.mark.asyncio
async def test_stored_file_is_owner_only(tmp_path: Path) -> None:
    """Header values are credentials; the file holding them is readable by its owner alone.

    Killed by: src/uclone_x/tools/mcp_manager.py :: os.chmod(tmp, 0o600)
    Becomes: os.chmod(tmp, 0o644)
    """
    manager, _ = _manager(tmp_path)
    await manager.add(_remote())
    mode = stat.S_IMODE((tmp_path / "mcp_servers.json").stat().st_mode)
    assert mode == 0o600
    await manager.close()


@pytest.mark.asyncio
async def test_duplicate_name_is_refused(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    await manager.add(_remote())
    with pytest.raises(DuplicateServerError):
        await manager.add(_remote())
    await manager.close()


@pytest.mark.asyncio
async def test_import_accepts_claude_desktop_snippet(tmp_path: Path) -> None:
    """A README's `mcpServers` snippet imports as it is; a bad entry is named, not fatal."""
    manager, _ = _manager(tmp_path)
    snippet = json.dumps(
        {
            "mcpServers": {
                "web": {"type": "http", "url": "https://mcp.example.test/mcp"},
                "legacy": {"type": "sse", "url": "https://old.example.test/sse"},
            }
        }
    )
    added, skipped = await manager.import_json(snippet)
    assert [a["name"] for a in added] == ["web"]
    assert skipped[0]["name"] == "legacy"
    assert "sse" in skipped[0]["reason"]
    with pytest.raises(ValueError, match="not valid JSON"):
        await manager.import_json("{nope")
    await manager.close()


def test_spec_reads_a_command_entry() -> None:
    spec = spec_from_entry(
        "files", {"command": "npx", "args": ["-y", "server-fs"], "env": {"K": "v"}}
    )
    assert spec.transport == "stdio"
    assert spec.args == ("-y", "server-fs")
    with pytest.raises(ValueError, match="command"):
        spec_from_entry("bad", {"args": ["x"]})


# ── HTTP routes ─────────────────────────────────────────────────────────────


@pytest.fixture
def api(tmp_path: Path) -> TestClient:
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "store")
    app.state.mcp_manager._client_factory = _factory()  # pyright: ignore[reportPrivateUsage]
    return TestClient(app, base_url="http://localhost")


def test_api_add_list_remove_round_trip(api: TestClient) -> None:
    body = {
        "name": "web",
        "transport": "http",
        "url": "https://mcp.example.test/mcp",
        "headers": {"Authorization": "Bearer tok-123"},
    }
    created = api.post("/api/mcp/servers", json=body)
    assert created.status_code == 201, created.text
    assert created.json()["status"] == "connected"
    assert api.post("/api/mcp/servers", json=body).status_code == 409
    listed = api.get("/api/mcp/servers").json()
    assert [s["name"] for s in listed["servers"]] == ["web"]
    # Clones see it: the persona editor offers the new tools by name.
    assert "web__search" in api.get("/api/personas").json()["available_tools"]
    assert api.delete("/api/mcp/servers/web").json() == {"status": "ok"}
    assert api.delete("/api/mcp/servers/web").status_code == 404


def test_api_never_echoes_credential_values(api: TestClient) -> None:
    """Only the names of headers and env variables leave the Core, never their values.

    Killed by: src/uclone_x/tools/mcp_manager.py :: "header_keys": sorted(spec.headers),
    Becomes: "header_keys": dict(spec.headers),
    """
    api.post(
        "/api/mcp/servers",
        json={
            "name": "web",
            "transport": "http",
            "url": "https://mcp.example.test/mcp",
            "headers": {"Authorization": "Bearer tok-SECRET-9"},
        },
    )
    for response in (api.get("/api/mcp/servers"), api.post("/api/mcp/servers/web/reconnect")):
        assert "tok-SECRET-9" not in response.text
    assert api.get("/api/mcp/servers").json()["servers"][0]["header_keys"] == ["Authorization"]


def test_api_refuses_an_invalid_server_with_its_reason(api: TestClient) -> None:
    response = api.post("/api/mcp/servers", json={"name": "a b", "transport": "stdio"})
    assert response.status_code == 400
    assert "letters, digits" in response.json()["detail"]
    response = api.post("/api/mcp/servers", json={"name": "x", "transport": "stdio"})
    assert response.status_code == 400
    assert "command" in response.json()["detail"]


def test_api_refuses_cross_origin_requests(api: TestClient) -> None:
    """Another site open in the browser must not start a local program or read the list.

    Killed by: src/uclone_x/ui/app.py :: _refuse_unless_local(request)  # another tab must not start a local process
    Becomes: pass
    """
    evil = {"Origin": "https://evil.example"}
    add = api.post(
        "/api/mcp/servers",
        json={"name": "evil", "transport": "stdio", "command": "/bin/sh"},
        headers=evil,
    )
    assert add.status_code == 403
    assert api.get("/api/mcp/servers", headers=evil).status_code == 403
    assert api.post("/api/mcp/servers/import", json={"json": "{}"}, headers=evil).status_code == 403
    assert api.get("/api/mcp/servers").json()["servers"] == []


def test_api_refuses_a_dns_rebinding_page(tmp_path: Path) -> None:
    """A page whose own domain now resolves to 127.0.0.1 sends matching Origin and Host.

    The origin check passes that request; only the Host name tells it apart. Bound to
    every interface, so the app-wide loopback guard (#1413) is not installed and this
    route's own check is the one under test: a dashboard shared on the network still
    starts programs only for this computer.

    Killed by: src/uclone_x/ui/app.py :: if _host_header_hostname(request.headers.get("host", "")) not in _LOOPBACK_HOSTS:
    Becomes: if False:
    """
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "store", bind_host="0.0.0.0")
    app.state.mcp_manager._client_factory = _factory()  # pyright: ignore[reportPrivateUsage]
    api = TestClient(app, base_url="http://localhost")
    rebound = {"Origin": "http://evil.example:8000", "Host": "evil.example:8000"}
    add = api.post(
        "/api/mcp/servers",
        json={"name": "evil", "transport": "stdio", "command": "/bin/sh"},
        headers=rebound,
    )
    assert add.status_code == 403
    assert "this computer" in add.json()["detail"]
    assert api.get("/api/mcp/servers").json()["servers"] == []


def test_header_with_a_line_break_is_refused_without_quoting_it() -> None:
    """A value h11 would refuse (and quote in its error) is refused first, by field name.

    Killed by: src/uclone_x/tools/mcp_manager.py :: if any(_CONTROL_CHARACTERS.search(text) for text in texts):
    Becomes: if False:
    """
    with pytest.raises(ValidationError) as caught:
        MCPServerSpec(
            name="web",
            transport="http",
            url="https://mcp.example.test/mcp",
            headers={"Authorization": "Bearer tok-SECRET\nX-Injected: 1"},
        )
    assert "header" in str(caught.value)
    assert "tok-SECRET" not in str(caught.value)


def test_pasted_header_value_is_trimmed() -> None:
    """A token copied with its trailing newline still works.

    Killed by: src/uclone_x/tools/mcp_manager.py :: return {k: v.strip() if isinstance(v, str) else v for k, v in items}
    Becomes: return {k: v for k, v in items}
    """
    spec = MCPServerSpec(
        name="web",
        transport="http",
        url="https://mcp.example.test/mcp",
        headers={"Authorization": "Bearer tok\n"},
    )
    assert spec.headers == {"Authorization": "Bearer tok"}


def test_view_hides_query_values_in_the_address(api: TestClient) -> None:
    """Some servers take their key in the address; the dashboard shows only its name.

    Killed by: src/uclone_x/tools/mcp_manager.py :: "url": _redact_query(spec.url),
    Becomes: "url": spec.url,
    """
    api.post(
        "/api/mcp/servers",
        json={
            "name": "web",
            "transport": "http",
            "url": "https://mcp.example.test/mcp?api_key=k-SECRET&region=eu",
        },
    )
    listed = api.get("/api/mcp/servers").text
    assert "k-SECRET" not in listed
    assert "api_key=...&region=..." in listed


def test_unreachable_address_error_hides_query_values() -> None:
    """The error shown for an address that does not answer repeats it, without its key.

    Killed by: src/uclone_x/tools/mcp_manager.py :: return f"Could not reach {_redact_query(spec.url)}. Check the address and your connection."
    Becomes: return f"Could not reach {spec.url}. Check the address and your connection."
    """
    from uclone_x.tools.mcp_manager import (
        _connect_error_text,  # pyright: ignore[reportPrivateUsage]
    )

    spec = MCPServerSpec(
        name="web", transport="http", url="https://mcp.example.test/mcp?api_key=k-SECRET"
    )
    text = _connect_error_text(spec, httpx.ConnectError("refused"))
    assert "k-SECRET" not in text
    assert "api_key=..." in text


_ECHO_SERVER = r"""
import json, os, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    if req["method"] == "initialize":
        result = {"protocolVersion": "2024-11-05"}
    elif req["method"] == "tools/list":
        result = {"tools": [{"name": "where", "description": "cwd and PATH"}]}
    else:
        text = f"{os.getcwd()}|{'PATH' in os.environ}|{os.environ.get('LEAK', 'absent')}"
        result = {"content": [{"type": "text", "text": text}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


@pytest.mark.asyncio
async def test_local_server_runs_in_the_workspace_with_a_narrow_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command server starts in the workspace, gets PATH, and not the rest of the host env.

    Killed by: src/uclone_x/tools/mcp_manager.py :: env_allowlist=LOCAL_SERVER_ENV_ALLOWLIST,
    Becomes: env_allowlist=LOCAL_SERVER_ENV_ALLOWLIST + ("LEAK",),
    """
    import sys

    from uclone_x.tools.models import ToolContext

    monkeypatch.setenv("LEAK", "host-secret")
    script = tmp_path / "echo_server.py"
    script.write_text(_ECHO_SERVER)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry()
    manager = MCPServerManager(
        registry=registry, config_path=tmp_path / "m.json", workspace_root=workspace
    )
    view = await manager.add(
        MCPServerSpec(name="local", transport="stdio", command=sys.executable, args=(str(script),))
    )
    assert view["status"] == "connected", view["error"]
    tool = registry.get("local__where")
    assert tool is not None
    ctx = ToolContext(agent_id="a", session_id="s", workspace_root=workspace)
    result = await tool.execute({}, ctx)
    output = result.output
    assert isinstance(output, list) and isinstance(output[0], dict), result.error
    text = str(output[0]["text"])
    cwd, has_path, leak = text.split("|")
    assert Path(cwd).resolve() == workspace.resolve()
    assert has_path == "True"
    assert leak == "absent"
    await manager.close()


@pytest.mark.asyncio
async def test_a_tool_held_across_removal_does_not_restart_its_server(tmp_path: Path) -> None:
    """A turn that looked the tool up before Remove must not start the program again.

    Killed by: src/uclone_x/tools/client.py :: self._retired = True
    Becomes: pass
    """
    import sys

    from uclone_x.tools.models import ToolContext

    script = tmp_path / "echo_server.py"
    script.write_text(_ECHO_SERVER)
    registry = ToolRegistry()
    manager = MCPServerManager(
        registry=registry, config_path=tmp_path / "m.json", workspace_root=tmp_path
    )
    await manager.add(
        MCPServerSpec(name="local", transport="stdio", command=sys.executable, args=(str(script),))
    )
    tool = registry.get("local__where")
    assert tool is not None
    await manager.remove("local")
    result = await tool.execute(
        {}, ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)
    )
    assert result.success is False
    assert "removed or turned off" in (result.error or "")
    await manager.close()
