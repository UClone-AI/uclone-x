"""`MCPServerManager` keeps its servers in an `MCPServerStoreProtocol` store (#1735).

The app still passes `config_path`, which is `FileMCPServerStore` over `mcp_servers.json`
(`tests/unit/test_mcp_manager.py` covers that file). These tests run the manager over
`InMemoryMCPServerStore`: it loads, connects, saves and removes servers, and writes
nothing to disk. Remote servers are an `httpx.MockTransport`; nothing leaves the process.
The last two pin that the file store still reports, in `load_error`, a file it cannot
read and an entry it cannot use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from uclone_x.tools.client import MCPClient
from uclone_x.tools.mcp_manager import InMemoryMCPServerStore, MCPServerManager, MCPServerSpec
from uclone_x.tools.models import MCPConnectionConfig
from uclone_x.tools.registry import ToolRegistry


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "DELETE":
        return httpx.Response(200)
    body: dict[str, Any] = json.loads(request.content)
    if "id" not in body:
        return httpx.Response(202)
    if body["method"] == "initialize":
        result: dict[str, Any] = {"protocolVersion": "2025-06-18"}
    elif body["method"] == "tools/list":
        result = {"tools": [{"name": "search", "description": "does search"}]}
    else:
        result = {"content": [{"type": "text", "text": "ok"}]}
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def _client(config: MCPConnectionConfig, prefix: str) -> MCPClient:
    return MCPClient(
        config=config, tool_name_prefix=prefix, http_transport=httpx.MockTransport(_handler)
    )


def _remote(name: str) -> MCPServerSpec:
    return MCPServerSpec(name=name, transport="http", url="https://mcp.example.test/mcp")


def _manager(
    store: InMemoryMCPServerStore, workspace: Path
) -> tuple[MCPServerManager, ToolRegistry]:
    registry = ToolRegistry()
    manager = MCPServerManager(
        registry=registry,
        config_path=None,
        workspace_root=workspace,
        client_factory=_client,
        store=store,
    )
    return manager, registry


@pytest.mark.asyncio
async def test_the_manager_saves_each_change_to_its_store_and_no_file(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/mcp_manager.py :: self._store.save({name: s.spec for name, s in self._servers.items()})
    Becomes: pass
    """
    store = InMemoryMCPServerStore()
    manager, registry = _manager(store, tmp_path)

    view = await manager.add(_remote("web"))
    assert view["status"] == "connected"
    assert [t.name for t in registry.list_tools()] == ["web__search"]
    assert list(store.servers) == ["web"]

    await manager.set_enabled("web", False)
    assert store.servers["web"].enabled is False

    await manager.remove("web")
    assert store.servers == {}
    assert manager.config_path is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_the_manager_starts_the_servers_its_store_holds(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/mcp_manager.py :: specs, self._load_error = self._store.load()
    Becomes: specs, self._load_error = {}, None
    """
    off = _remote("off").model_copy(update={"enabled": False})
    store = InMemoryMCPServerStore({"web": _remote("web"), "off": off})
    manager, registry = _manager(store, tmp_path)

    await manager.start()

    assert {v["name"]: v["status"] for v in manager.views()} == {
        "web": "connected",
        "off": "disabled",
    }
    assert [t.name for t in registry.list_tools()] == ["web__search"]
    assert manager.load_error is None
    await manager.close()


def test_a_manager_takes_a_config_path_or_a_store_not_both(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/mcp_manager.py :: raise ValueError("Give an MCPServerManager a config_path or a store, exactly one")
    Becomes: pass
    """
    with pytest.raises(ValueError, match="exactly one"):
        MCPServerManager(
            registry=ToolRegistry(),
            config_path=tmp_path / "mcp_servers.json",
            workspace_root=tmp_path,
            store=InMemoryMCPServerStore(),
        )


def test_the_file_store_reports_a_file_it_cannot_read(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/mcp_manager.py :: return servers, f"{self._path} could not be read: {err}"
    Becomes: return servers, None
    """
    path = tmp_path / "mcp_servers.json"
    path.write_text("{not json", encoding="utf-8")

    manager = MCPServerManager(registry=ToolRegistry(), config_path=path, workspace_root=tmp_path)

    assert manager.config_path == path
    assert manager.views() == []
    assert str(manager.load_error).startswith(f"{path} could not be read: ")


def test_the_file_store_keeps_valid_servers_and_names_an_invalid_one(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/mcp_manager.py :: load_error = f"Server '{name}' in {self._path} is invalid: {err}"
    Becomes: pass
    """
    path = tmp_path / "mcp_servers.json"
    entries = {"web": {"url": "https://mcp.example.test/mcp"}, "bad": {"command": ""}}
    path.write_text(json.dumps({"mcpServers": entries}), encoding="utf-8")

    manager = MCPServerManager(registry=ToolRegistry(), config_path=path, workspace_root=tmp_path)

    assert [v["name"] for v in manager.views()] == ["web"]
    assert str(manager.load_error).startswith(f"Server 'bad' in {path} is invalid: ")
