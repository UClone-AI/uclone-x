"""Unit tests for dynamic MCP configuration loader and tool registry integration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.sandbox.models import (
    ContainerIsolation,
    IsolationLevel,
    NoIsolation,
    WasmIsolation,
    WorkspaceIsolation,
)
from uclone_x.tools.builtin.mcp_loader import (
    MCPConfigFileLoader,
    expand_env_vars,
)
from uclone_x.tools.client import MCPClient, MCPTool
from uclone_x.tools.models import MCPConnectionConfig, MCPTransport, ToolContext, ToolResult
from uclone_x.tools.protocols import MCPClientProtocol
from uclone_x.tools.registry import ToolRegistry, create_default_registry

MOCK_MCP_SERVER_CODE = """
import sys, json

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line.strip())
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock_loader_server", "version": "1.0.0"}
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "dynamic_echo",
                        "description": "Echoes text dynamically",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"msg": {"type": "string"}},
                            "required": ["msg"]
                        }
                    },
                    {
                        "name": "dynamic_calc",
                        "description": "Calculates sum",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"a": {"type": "number"}, "b": {"type": "number"}}
                        }
                    }
                ]
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "dynamic_echo":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": args.get("msg", "")}], "isError": False}
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": "result"}], "isError": False}
            }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
"""


# ======================================================================================
# 1. Environment Variable Expansion Tests
# ======================================================================================


def test_expand_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """expand_env_vars correctly expands ${VAR}, ${VAR:-default}, and $VAR patterns."""
    monkeypatch.setenv("TEST_HOST", "localhost")
    monkeypatch.setenv("TEST_PORT", "9000")
    monkeypatch.delenv("UNSET_VAR", raising=False)

    assert expand_env_vars("http://${TEST_HOST}:${TEST_PORT}/mcp") == "http://localhost:9000/mcp"
    assert expand_env_vars("http://${TEST_HOST}:$TEST_PORT/mcp") == "http://localhost:9000/mcp"
    assert expand_env_vars("${UNSET_VAR:-default_val}") == "default_val"
    assert expand_env_vars("${UNSET_VAR}") == ""
    assert expand_env_vars("$UNSET_VAR") == ""
    assert expand_env_vars("plain_string") == "plain_string"


def test_substitute_env_recursive(monkeypatch: pytest.MonkeyPatch) -> None:
    """_substitute_env recursively processes nested dicts and lists."""
    monkeypatch.setenv("APP_NAME", "my_service")
    monkeypatch.setenv("API_URL", "https://api.example.com")

    raw = {
        "name": "${APP_NAME}",
        "endpoints": ["${API_URL}/v1", "$API_URL/v2"],
        "nested": {"key": "${APP_NAME}_key", "num": 123},
    }

    processed = MCPConfigFileLoader.substitute_env(raw)
    assert processed["name"] == "my_service"
    assert processed["endpoints"] == ["https://api.example.com/v1", "https://api.example.com/v2"]
    assert processed["nested"]["key"] == "my_service_key"
    assert processed["nested"]["num"] == 123


# ======================================================================================
# 2. Config Discovery Tests
# ======================================================================================


def test_discover_config_file_explicit(tmp_path: Path) -> None:
    """Explicit config_path is returned if existing."""
    cfg_file = tmp_path / "custom_mcp.json"
    cfg_file.write_text('{"mcpServers": {}}')

    loader = MCPConfigFileLoader(config_path=cfg_file)
    assert loader.discover_config_file() == cfg_file

    missing_loader = MCPConfigFileLoader(config_path=tmp_path / "missing.json")
    assert missing_loader.discover_config_file() is None


def test_discover_config_file_workspace_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loader discovers .uclone/mcp.json or mcp.json in workspace_root or cwd."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    dot_uclone = ws / ".uclone"
    dot_uclone.mkdir()
    ws_cfg = dot_uclone / "mcp.json"
    ws_cfg.write_text('{"mcpServers": {}}')

    loader = MCPConfigFileLoader(workspace_root=ws)
    assert loader.discover_config_file() == ws_cfg

    # Root mcp.json fallback
    ws_cfg.unlink()
    root_cfg = ws / "mcp.json"
    root_cfg.write_text('{"mcpServers": {}}')
    assert loader.discover_config_file() == root_cfg

    # Cwd discovery
    root_cfg.unlink()
    monkeypatch.chdir(ws)
    loader_cwd = MCPConfigFileLoader()
    ws_cwd_cfg = ws / ".uclone" / "mcp.json"
    ws_cwd_cfg.write_text('{"mcpServers": {}}')
    assert loader_cwd.discover_config_file() == ws_cwd_cfg


def test_discover_config_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None when no config file exists in candidate locations."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    monkeypatch.setenv("HOME", str(empty_dir))

    loader = MCPConfigFileLoader(workspace_root=empty_dir)
    assert loader.discover_config_file() is None
    assert loader.load_configs() == []


# ======================================================================================
# 3. Config Parsing Formats and Validations
# ======================================================================================


def test_parse_standard_mcp_servers_format(tmp_path: Path) -> None:
    """Parse standard mcpServers schema with stdio and network configurations."""
    json_content = json.dumps(
        {
            "mcpServers": {
                "git_server": {
                    "command": "npx",
                    "args": ["-y", "mcp-git"],
                    "env": {"GIT_AUTHOR": "Agent"},
                    "env_allowlist": ["PATH", "HOME"],
                    "workspace_root": str(tmp_path),
                },
                "remote_sse": {
                    "transport": "sse",
                    "url": "https://api.example.com/sse",
                    "allow_network": True,
                    "egress_allowlist": ["api.example.com"],
                },
            }
        }
    )

    loader = MCPConfigFileLoader(workspace_root=tmp_path)
    configs = loader.parse_config_json(json_content)

    assert len(configs) == 2
    git_cfg = next(c for c in configs if c.server_name == "git_server")
    assert git_cfg.command == "npx"
    assert git_cfg.args == ("-y", "mcp-git")
    assert git_cfg.env == {"GIT_AUTHOR": "Agent"}
    assert git_cfg.env_allowlist == ("PATH", "HOME")
    assert git_cfg.transport is MCPTransport.STDIO
    assert git_cfg.workspace_root == tmp_path
    assert git_cfg.isolation.level is IsolationLevel.WORKSPACE

    sse_cfg = next(c for c in configs if c.server_name == "remote_sse")
    assert sse_cfg.transport is MCPTransport.SSE
    assert sse_cfg.url == "https://api.example.com/sse"
    assert sse_cfg.effective_allow_network is True
    assert sse_cfg.effective_egress_allowlist == ("api.example.com",)
    assert sse_cfg.isolation.level is IsolationLevel.NONE


def test_parse_claude_desktop_type_key_and_headers(tmp_path: Path) -> None:
    """A snippet a vendor publishes names the transport `type` and carries sign-in headers.

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: transport_raw = raw_cfg.get("transport", raw_cfg.get("type"))
    Becomes: transport_raw = raw_cfg.get("transport")
    """
    json_content = json.dumps(
        {
            "mcpServers": {
                "remote": {
                    "type": "http",
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer t"},
                    "allow_network": True,
                },
            }
        }
    )
    (cfg,) = MCPConfigFileLoader(workspace_root=tmp_path).parse_config_json(json_content)
    assert cfg.transport is MCPTransport.STREAMABLE_HTTP
    assert cfg.headers == {"Authorization": "Bearer t"}


def test_parse_direct_dictionary_and_list_formats(tmp_path: Path) -> None:
    """Parse direct dictionary mapping and list-based server configurations."""
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    # Direct mapping
    direct_dict = {
        "server_a": {"command": "python", "args": ["a.py"]},
        "server_b": {"command": "node", "args": ["b.js"]},
    }
    configs_dict = loader.parse_config_dict(direct_dict)
    assert len(configs_dict) == 2
    assert {c.server_name for c in configs_dict} == {"server_a", "server_b"}

    # List format
    list_data = [
        {"server_name": "srv1", "command": "python"},
        {"name": "srv2", "url": "https://srv2.internal/mcp", "transport": "sse"},
    ]
    configs_list = loader.parse_config_dict(list_data)
    assert len(configs_list) == 2
    assert configs_list[0].server_name == "srv1"
    assert configs_list[1].server_name == "srv2"
    assert configs_list[1].transport is MCPTransport.SSE


def test_parse_isolation_types(tmp_path: Path) -> None:
    """Loader correctly maps isolation string options to IsolationPolicy instances."""
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    configs = loader.parse_config_dict(
        {
            "s_none": {"command": "echo", "isolation": "none"},
            "s_ws": {"command": "echo", "isolation": "workspace"},
            "s_container": {
                "command": "echo",
                "isolation": "container",
                "allow_network": True,
                "egress_allowlist": ["api.github.com"],
            },
            "s_wasm": {"command": "echo", "isolation": "wasm", "allow_network": False},
        }
    )

    by_name = {c.server_name: c for c in configs}
    assert isinstance(by_name["s_none"].isolation, NoIsolation)
    assert isinstance(by_name["s_ws"].isolation, WorkspaceIsolation)
    assert isinstance(by_name["s_container"].isolation, ContainerIsolation)
    assert by_name["s_container"].effective_allow_network is True
    assert isinstance(by_name["s_wasm"].isolation, WasmIsolation)


def test_parse_invalid_json_and_schema_errors(tmp_path: Path) -> None:
    """Invalid JSON syntax, non-existent files, and bad configurations raise clear ValueErrors."""
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    # Missing file
    with pytest.raises(FileNotFoundError, match="MCP configuration file not found"):
        loader.load_from_file(tmp_path / "non_existent.json")

    # Invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.parse_config_json("{ bad json }")

    # Invalid root type
    with pytest.raises(ValueError, match="expected dict or list"):
        loader.parse_config_dict("a string is not a config")

    # Invalid transport
    with pytest.raises(ValueError, match="Unknown MCP transport 'invalid_transport'"):
        loader.parse_config_dict({"srv": {"command": "ls", "transport": "invalid_transport"}})

    # Unsupported isolation
    with pytest.raises(ValueError, match="Unsupported isolation type 'hypervisor'"):
        loader.parse_config_dict({"srv": {"command": "ls", "isolation": "hypervisor"}})


# ======================================================================================
# 4. Client Instantiation and Tool Registration Tests
# ======================================================================================


def test_create_clients(tmp_path: Path) -> None:
    """create_clients returns MCPClient instances for configs."""
    loader = MCPConfigFileLoader(workspace_root=tmp_path)
    cfg = MCPConnectionConfig(
        server_name="test_client_gen",
        command="python",
        workspace_root=tmp_path,
    )
    clients = loader.create_clients([cfg])
    assert len(clients) == 1
    assert isinstance(clients[0], MCPClient)
    assert clients[0].config.server_name == "test_client_gen"


@pytest.mark.asyncio
async def test_register_client_tools_async_with_mock() -> None:
    """register_client_tools_async discovers and registers tools from an MCPClient."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader()

    mock_client = MagicMock(spec=MCPClientProtocol)
    mock_client.config = MCPConnectionConfig(
        server_name="mock_srv",
        command="python",
        workspace_root=Path("/tmp"),
    )

    t1 = MCPTool("t1", "desc1", {}, mock_client)
    t2 = MCPTool("t2", "desc2", {}, mock_client)
    mock_client.list_tools = AsyncMock(return_value=[t1, t2])

    registered = await loader.register_client_tools_async(registry, mock_client)
    assert len(registered) == 2
    assert registry.get("t1") is t1
    assert registry.get("t2") is t2


@pytest.mark.asyncio
async def test_register_client_tools_async_handles_inaccessible_server() -> None:
    """register_client_tools_async gracefully catches exceptions and logs warnings without crashing."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader()

    broken_client = MagicMock(spec=MCPClientProtocol)
    broken_client.config = MCPConnectionConfig(
        server_name="broken_srv",
        command="missing_binary",
        workspace_root=Path("/tmp"),
    )
    broken_client.list_tools = AsyncMock(side_effect=FileNotFoundError("binary not found"))

    registered = await loader.register_client_tools_async(registry, broken_client)
    assert registered == []
    assert len(registry.list_tools()) == 0


def test_register_client_tools_sync_both_modes(tmp_path: Path) -> None:
    """register_client_tools works synchronously with or without a running event loop."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    mock_client = MagicMock(spec=MCPClientProtocol)
    mock_client.config = MCPConnectionConfig(
        server_name="sync_srv",
        command="python",
        workspace_root=tmp_path,
    )
    t = MCPTool("sync_tool", "desc", {}, mock_client)
    mock_client.list_tools = AsyncMock(return_value=[t])

    # 1. Sync invocation in normal thread
    res = loader.register_client_tools(registry, mock_client)
    assert len(res) == 1
    assert registry.get("sync_tool") is t


@pytest.mark.asyncio
async def test_register_client_tools_sync_inside_running_loop(tmp_path: Path) -> None:
    """register_client_tools works synchronously even when called from an active event loop."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    mock_client = MagicMock(spec=MCPClientProtocol)
    mock_client.config = MCPConnectionConfig(
        server_name="thread_srv",
        command="python",
        workspace_root=tmp_path,
    )
    t = MCPTool("thread_tool", "desc", {}, mock_client)
    mock_client.list_tools = AsyncMock(return_value=[t])

    # Called inside async function with active event loop
    res = loader.register_client_tools(registry, mock_client)
    assert len(res) == 1
    assert registry.get("thread_tool") is t


@pytest.mark.asyncio
async def test_load_and_register_tools_sync_inside_running_loop(tmp_path: Path) -> None:
    """load_and_register_tools works synchronously when called from an active event loop."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    cfg = MCPConnectionConfig(
        server_name="loop_srv",
        command=sys.executable,
        args=("-c", MOCK_MCP_SERVER_CODE),
        workspace_root=tmp_path,
    )

    # Empty configs returns []
    assert loader.load_and_register_tools(registry, []) == []

    # Non-empty configs
    registered = loader.load_and_register_tools(registry, [cfg])
    assert len(registered) == 2
    assert registry.get("dynamic_echo") is not None


def test_mcp_loader_properties_and_alternate_formats(tmp_path: Path) -> None:
    """Test loader properties, alternate 'servers' key, and skip non-dict items."""
    loader = MCPConfigFileLoader(workspace_root=tmp_path, config_path=tmp_path / "custom.json")
    assert loader.workspace_root == tmp_path
    assert loader.config_path == tmp_path / "custom.json"

    # 'servers' alternate key
    cfgs = loader.parse_config_dict(
        {"servers": {"s1": {"command": "python"}, "invalid_skipped": "not_a_dict"}}
    )
    assert len(cfgs) == 1
    assert cfgs[0].server_name == "s1"

    # List format with non-dict item skipped
    list_cfgs = loader.parse_config_dict(
        [{"server_name": "s_list", "command": "python"}, "not_a_dict_skipped"]
    )
    assert len(list_cfgs) == 1
    assert list_cfgs[0].server_name == "s_list"

    # Direct IsolationPolicy object passed
    direct_iso_cfgs = loader.parse_config_dict(
        {"s_direct": {"command": "python", "isolation": NoIsolation()}}
    )
    assert isinstance(direct_iso_cfgs[0].isolation, NoIsolation)

    # Invalid isolation type (non-string, non-IsolationPolicy)
    with pytest.raises(ValueError, match="Invalid isolation specification"):
        loader.parse_config_dict({"s_err": {"command": "python", "isolation": 12345}})


def test_expand_env_vars_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """expand_env_vars with empty variable falls back to default if provided."""
    monkeypatch.setenv("EMPTY_VAR", "")
    assert expand_env_vars("${EMPTY_VAR:-fallback}") == "fallback"


@pytest.mark.asyncio
async def test_load_and_register_tools_async_multiple_servers(tmp_path: Path) -> None:
    """load_and_register_tools_async handles multiple servers, skipping broken ones."""
    registry = ToolRegistry()
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    cfg_good = MCPConnectionConfig(
        server_name="good_srv",
        command=sys.executable,
        args=("-c", MOCK_MCP_SERVER_CODE),
        workspace_root=tmp_path,
    )
    cfg_broken = MCPConnectionConfig(
        server_name="bad_srv",
        command="non_existent_command_xyz_123",
        workspace_root=tmp_path,
    )

    registered = await loader.load_and_register_tools_async(registry, [cfg_good, cfg_broken])
    assert len(registered) == 2
    tool_names = {t.name for t in registered}
    assert "dynamic_echo" in tool_names
    assert "dynamic_calc" in tool_names

    # Verify tool execution on registry
    echo_tool = registry.get("dynamic_echo")
    assert echo_tool is not None
    ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
    res = await echo_tool.execute({"msg": "hello from dynamic MCP"}, ctx)
    assert res.success is True
    assert res.output == [{"type": "text", "text": "hello from dynamic MCP"}]


def test_load_and_register_tools_sync_end_to_end(tmp_path: Path) -> None:
    """load_and_register_tools synchronously loads config file and auto-registers MCP tools."""
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "test_server": {
                        "command": sys.executable,
                        "args": ["-c", MOCK_MCP_SERVER_CODE],
                        "workspace_root": str(tmp_path),
                    }
                }
            }
        )
    )

    registry = ToolRegistry()
    loader = MCPConfigFileLoader(workspace_root=tmp_path, config_path=cfg_file)
    registered = loader.load_and_register_tools(registry)

    assert len(registered) == 2
    assert registry.get("dynamic_echo") is not None
    assert registry.get("dynamic_calc") is not None


def test_mcp_loader_omitted_workspace_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test that MCPConfigFileLoader correctly falls back to Path.cwd() when
    workspace_root is omitted in the config file, ensuring it honors the
    WorkspaceIsolation guarantee for STDIO transports.

    Runs in its own directory: in the directory pytest was started from, the repo-wide
    `_isolate_mcp_config_discovery` fixture answers the loader's `cwd()` with a stand-in.

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: ws_root = Path.cwd()
    """
    from uclone_x.sandbox.models import IsolationLevel
    from uclone_x.tools.builtin.mcp_loader import MCPConfigFileLoader
    from uclone_x.tools.models import MCPTransport

    monkeypatch.chdir(tmp_path)
    loader = MCPConfigFileLoader()
    configs = loader.parse_config_dict({"servers": {"test_omitted": {"command": "/bin/true"}}})
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.transport is MCPTransport.STDIO
    assert cfg.isolation.level is IsolationLevel.WORKSPACE
    assert cfg.workspace_root == Path.cwd()


def test_mcp_transport_members_and_values_regression_579() -> None:
    """Assert MCPTransport members and values, ensuring no same-valued alias exists (Issue #579).

    Killed by: src/uclone_x/tools/models.py :: STREAMABLE_HTTP = "http"
    """
    assert MCPTransport.STDIO == "stdio"
    assert MCPTransport.SSE == "sse"
    assert MCPTransport.STREAMABLE_HTTP == "http"
    assert MCPTransport.WEBSOCKET == "websocket"

    # Verify no same-valued alias exists and member count is 4
    members = list(MCPTransport)
    assert len(members) == 4
    assert len({m.value for m in members}) == 4

    # Verify HTTP_SSE is no longer a member or attribute of MCPTransport
    assert not hasattr(MCPTransport, "HTTP_SSE")

    # Verify lookup by value
    assert MCPTransport("http") is MCPTransport.STREAMABLE_HTTP
    assert MCPTransport("sse") is MCPTransport.SSE
    assert MCPTransport("stdio") is MCPTransport.STDIO
    assert MCPTransport("websocket") is MCPTransport.WEBSOCKET


def test_mcp_loader_transport_inference_url_without_command_regression_579(tmp_path: Path) -> None:
    """Assert transport inference behavior for url without command (Issue #579).

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: elif url_raw is not None and command_raw is None:
    """
    loader = MCPConfigFileLoader(workspace_root=tmp_path)

    # Case 1: url provided without command and without transport -> infers SSE
    configs_sse = loader.parse_config_dict(
        {
            "inferred_sse": {
                "url": "https://api.example.com/events",
            }
        }
    )
    assert len(configs_sse) == 1
    assert configs_sse[0].server_name == "inferred_sse"
    assert configs_sse[0].transport is MCPTransport.SSE
    assert configs_sse[0].url == "https://api.example.com/events"
    assert configs_sse[0].command is None

    # Case 2: url provided with explicit streamable HTTP transport -> resolves to STREAMABLE_HTTP
    configs_http = loader.parse_config_dict(
        {
            "streamable_server": {
                "url": "https://api.example.com/mcp",
                "transport": "http",
            }
        }
    )
    assert len(configs_http) == 1
    assert configs_http[0].server_name == "streamable_server"
    assert configs_http[0].transport is MCPTransport.STREAMABLE_HTTP
    assert configs_http[0].url == "https://api.example.com/mcp"

    # Case 2b: "streamable_http" is also accepted as STREAMABLE_HTTP
    configs_streamable = loader.parse_config_dict(
        {
            "streamable_alias": {
                "url": "https://api.example.com/mcp",
                "transport": "streamable_http",
            }
        }
    )
    assert configs_streamable[0].transport is MCPTransport.STREAMABLE_HTTP

    # Case 3: command provided (even if url is present) without transport -> defaults to STDIO
    configs_stdio = loader.parse_config_dict(
        {
            "stdio_with_url": {
                "command": "python",
                "url": "https://api.example.com/ignored",
            }
        }
    )
    assert len(configs_stdio) == 1
    assert configs_stdio[0].transport is MCPTransport.STDIO
    assert configs_stdio[0].command == "python"


# ======================================================================================
# #1414: a file-loaded server's tools work from the loop that later calls them
# ======================================================================================


def _write_pid_reporting_server(tmp_path: Path) -> tuple[Path, Path]:
    """An `mcpServers` file for the mock server, which also appends its pid to a file."""
    pid_file = tmp_path / "server_pids.txt"
    script = tmp_path / "pid_server.py"
    script.write_text(
        "import os\n"
        f"with open({str(pid_file)!r}, 'a') as fh:\n"
        "    fh.write(f'{os.getpid()}\\n')\n" + MOCK_MCP_SERVER_CODE
    )
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "file_srv": {
                        "command": sys.executable,
                        "args": [str(script)],
                        "workspace_root": str(tmp_path),
                    }
                }
            }
        )
    )
    return cfg_file, pid_file


def _server_pids(pid_file: Path) -> list[int]:
    return [int(line) for line in pid_file.read_text().split()] if pid_file.exists() else []


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_file_loaded_tool_answers_from_a_later_loop(tmp_path: Path) -> None:
    """Registered from sync code, the tool is called in a new loop and answers (#1414).

    The registry is built inside a loop that closes when registration returns; the call
    happens in another. A client still holding the first loop's pipes cannot answer.

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: await client.disconnect()
    Becomes: pass
    """
    cfg_file, _pid_file = _write_pid_reporting_server(tmp_path)
    registry = create_default_registry(
        workspace_root=tmp_path, enable_mcp=True, mcp_config_path=cfg_file
    )
    tool = registry.get("dynamic_echo")
    assert isinstance(tool, MCPTool)

    async def call_then_close() -> ToolResult:
        ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
        try:
            return await tool.execute({"msg": "from a later loop"}, ctx)
        finally:
            await cast(MCPClient, tool._client).disconnect()  # pyright: ignore[reportPrivateUsage]

    result = asyncio.run(call_then_close())

    assert result.error is None
    assert result.success is True
    assert result.output == [{"type": "text", "text": "from a later loop"}]


async def test_a_file_loaded_tool_answers_when_registered_inside_a_running_loop(
    tmp_path: Path,
) -> None:
    """Registered from a running loop (the worker-thread branch), it answers on that loop.

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: await client.disconnect()
    Becomes: pass
    """
    cfg_file, _pid_file = _write_pid_reporting_server(tmp_path)
    registry = create_default_registry(
        workspace_root=tmp_path, enable_mcp=True, mcp_config_path=cfg_file
    )
    tool = registry.get("dynamic_echo")
    assert isinstance(tool, MCPTool)

    ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
    try:
        result = await asyncio.wait_for(tool.execute({"msg": "same loop"}, ctx), timeout=10.0)
    finally:
        await cast(MCPClient, tool._client).disconnect()  # pyright: ignore[reportPrivateUsage]

    assert result.error is None
    assert result.success is True
    assert result.output == [{"type": "text", "text": "same loop"}]


def test_registration_leaves_no_server_running_and_a_call_starts_one(tmp_path: Path) -> None:
    """The server that answered the listing is stopped; the first call starts a fresh one.

    Killed by: src/uclone_x/tools/builtin/mcp_loader.py :: await client.disconnect()
    Becomes: pass
    """
    cfg_file, pid_file = _write_pid_reporting_server(tmp_path)
    registry = create_default_registry(
        workspace_root=tmp_path, enable_mcp=True, mcp_config_path=cfg_file
    )
    tool = registry.get("dynamic_echo")
    assert isinstance(tool, MCPTool)
    client = cast(MCPClient, tool._client)  # pyright: ignore[reportPrivateUsage]

    [listing_pid] = _server_pids(pid_file)
    assert not client.is_connected
    assert not _is_alive(listing_pid)

    async def call_then_close() -> None:
        ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
        result = await tool.execute({"msg": "x"}, ctx)
        assert result.success is True
        await client.disconnect()

    asyncio.run(call_then_close())

    pids = _server_pids(pid_file)
    assert len(pids) == 2
    assert not any(_is_alive(pid) for pid in pids)
