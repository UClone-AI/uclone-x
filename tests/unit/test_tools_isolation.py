"""Unit tests for tool isolation boundaries, environment allowlists, and MCP integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import uclone_x
from uclone_x.core.provenance import Provenance
from uclone_x.errors import PathTraversalError, SandboxViolationError
from uclone_x.sandbox.models import (
    ContainerIsolation,
    IsolationLevel,
    NoIsolation,
    is_secret_env_name,
)
from uclone_x.tools import (
    LocalTool,
    MCPClient,
    MCPConnectionConfig,
    MCPTool,
    MCPTransport,
    ToolContext,
    ToolRegistry,
    ToolResult,
)

# A minimal in-line Python script that acts as an MCP stdio server for testing.
MCP_MOCK_SERVER_SCRIPT = """
import sys, json, os

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
                "serverInfo": {"name": "test_mcp_server", "version": "1.0.0"}
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
                        "name": "echo",
                        "description": "Echoes text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "env_inspector",
                        "description": "Inspects environment variables",
                        "inputSchema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "file_reader",
                        "description": "Reads a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"]
                        }
                    },
                    {
                        "name": "error_trigger",
                        "description": "Triggers an error",
                        "inputSchema": {"type": "object", "properties": {}}
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

        if tool_name == "echo":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}],
                    "isError": False
                }
            }
        elif tool_name == "env_inspector":
            data = {
                "SAFE": os.getenv("SAFE_VAR"),
                "KEY": os.getenv("OPENAI_API_KEY"),
                "AWS": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "EXPLICIT_KEY": os.getenv("EXPLICIT_KEY"),
                "UNALLOWLISTED": os.getenv("UNALLOWLISTED_HOST_VAR"),
            }
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": json.dumps(data),
                    "isError": False
                }
            }
        elif tool_name == "file_reader":
            fp = args.get("file_path", "")
            try:
                with open(fp, "r") as f:
                    content = f.read()
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": content}],
                        "isError": False
                    }
                }
            except Exception as e:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": str(e),
                        "isError": True
                    }
                }
        elif tool_name == "error_trigger":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": "Custom internal error"}
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
"""


# ======================================================================================
# 1. Models and Configuration Tests
# ======================================================================================


def test_tool_context_requires_workspace_root() -> None:
    """ToolContext defaults workspace_root to None and requires it explicitly via require_workspace."""
    ctx_none = ToolContext.model_validate({"agent_id": "a1", "session_id": "s1"})
    assert ctx_none.workspace_root is None
    with pytest.raises(
        RuntimeError, match="Tool requires a workspace, but the host provided none."
    ):
        ctx_none.require_workspace()

    ctx = ToolContext(
        agent_id="a1",
        session_id="s1",
        workspace_root=Path("/safe/workspace"),
    )
    assert ctx.workspace_root == Path("/safe/workspace")
    assert ctx.isolation.level is IsolationLevel.WORKSPACE
    assert ctx.timeout_seconds == 30.0


def test_mcp_connection_config_defaults_and_fields(tmp_path: Path) -> None:
    """MCPConnectionConfig carries isolation, env_allowlist, workspace_root, and egress fields."""
    cfg = MCPConnectionConfig(
        server_name="git_server",
        command="npx",
        args=("-y", "mcp-git"),
        workspace_root=tmp_path,
        env_allowlist=("PATH", "HOME"),
        allow_network=False,
        egress_allowlist=("api.github.com",),
    )
    assert cfg.server_name == "git_server"
    assert cfg.transport is MCPTransport.STDIO
    assert cfg.isolation.level is IsolationLevel.WORKSPACE
    assert cfg.workspace_root == tmp_path
    assert cfg.env_allowlist == ("PATH", "HOME")
    assert cfg.effective_allow_network is False
    assert cfg.effective_egress_allowlist == ("api.github.com",)

    # Issue #34: ContainerIsolation is the single source of truth for network policy
    container_cfg = MCPConnectionConfig(
        server_name="container_mcp",
        isolation=ContainerIsolation(
            image="node:20",
            allow_network=True,
            egress_allowlist=("api.github.com", "*.npmjs.org"),
        ),
        allow_network=True,
    )
    assert container_cfg.effective_allow_network is True
    assert container_cfg.effective_egress_allowlist == ("api.github.com", "*.npmjs.org")


def test_tool_result_unifies_isolation_and_provenance() -> None:
    """ToolResult includes isolation_level and in-band provenance."""
    prov = Provenance.primary(provider="mcp.server", model="echo")
    res = ToolResult(
        success=True,
        output="ok",
        execution_time_ms=12.5,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=prov,
    )
    assert res.success is True
    assert res.output == "ok"
    assert res.isolation_level is IsolationLevel.WORKSPACE
    assert res.provenance is prov


def test_is_secret_env_name_helper() -> None:
    """Secret naming patterns are identified by is_secret_env_name."""
    assert is_secret_env_name("OPENAI_API_KEY") is True
    assert is_secret_env_name("GH_TOKEN") is True
    assert is_secret_env_name("GITHUB_TOKEN") is True
    assert is_secret_env_name("AWS_SECRET_ACCESS_KEY") is True
    assert is_secret_env_name("DB_PASSWORD") is True
    assert is_secret_env_name("APP_CREDENTIALS") is True
    assert is_secret_env_name("API_SECRET") is True
    assert is_secret_env_name("PATH") is False
    assert is_secret_env_name("HOME") is False
    assert is_secret_env_name("LANG") is False


# ======================================================================================
# 2. LocalTool and ToolRegistry Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_local_tool_default_and_custom_execution(tmp_path: Path) -> None:
    ctx = ToolContext(
        agent_id="a1",
        session_id="s1",
        workspace_root=tmp_path,
    )

    # Tool with no handler raises RuntimeError (P6 / Issue #43)
    tool = LocalTool(name="default_tool", description="Default description")
    assert tool.name == "default_tool"
    assert tool.description == "Default description"
    assert tool.parameters_schema == {"type": "object", "properties": {}}

    with pytest.raises(RuntimeError, match="has no execution handler configured"):
        await tool.execute({}, ctx)

    # Custom execution handler
    async def custom_handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output={"doubled": params["x"] * 2},
            isolation_level=context.isolation.level,
            provenance=Provenance.primary("custom_tool"),
        )

    calc_tool = LocalTool(
        name="calc",
        description="Calculates",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=custom_handler,
    )
    res_calc = await calc_tool.execute({"x": 21}, ctx)
    assert res_calc.success is True
    assert res_calc.output == {"doubled": 42}


def test_tool_registry_management() -> None:
    registry = ToolRegistry()
    t1 = LocalTool(name="tool1", description="desc1")
    t2 = LocalTool(name="tool2", description="desc2")

    registry.register(t1)
    registry.register(t2)

    assert registry.get("tool1") is t1
    assert registry.get("tool2") is t2
    assert registry.get("missing") is None

    all_tools = registry.list_tools()
    assert len(all_tools) == 2

    filtered = registry.list_tools(filter_names=("tool2", "nonexistent"))
    assert len(filtered) == 1
    assert filtered[0] is t2


# ======================================================================================
# 3. MCP Subprocess Isolation & Secret Filtering Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_mcp_client_environment_scrubbing_and_secret_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP stdio subprocess must scrub host secrets from allowlist and omit unallowlisted vars."""
    monkeypatch.setenv("SAFE_VAR", "safe_val")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-test")
    monkeypatch.setenv("UNALLOWLISTED_HOST_VAR", "unallowlisted_val")

    config = MCPConnectionConfig(
        server_name="test_server",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-c", MCP_MOCK_SERVER_SCRIPT),
        workspace_root=tmp_path,
        env_allowlist=("SAFE_VAR",),
        env={"EXPLICIT_KEY": "sk-explicit-granted"},
    )

    async with MCPClient(config=config) as client:
        ctx = ToolContext(
            agent_id="a1",
            session_id="s1",
            workspace_root=tmp_path,
        )

        res = await client.call_tool("env_inspector", {}, ctx)
        assert res.success is True
        assert res.output is not None

        env_data = json.loads(str(res.output))
        # Allowlisted safe variable is present
        assert env_data["SAFE"] == "safe_val"
        # Unallowlisted host variable is omitted
        assert env_data["UNALLOWLISTED"] is None
        # Explicit variable passed in env is present
        assert env_data["EXPLICIT_KEY"] == "sk-explicit-granted"

    # Issue #38: Credential-shaped env_allowlist entry raises ValidationError
    with pytest.raises(
        ValidationError,
        match="Credential-shaped environment variable 'OPENAI_API_KEY' in env_allowlist is denied",
    ):
        MCPConnectionConfig(
            server_name="test_server",
            command="python",
            workspace_root=tmp_path,
            env_allowlist=("OPENAI_API_KEY",),
        )


@pytest.mark.asyncio
async def test_mcp_client_tools_list_and_call(tmp_path: Path) -> None:
    """Test discovery and invocation of tools via MCP JSON-RPC protocol."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("MCP workspace content")

    config = MCPConnectionConfig(
        server_name="test_server",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-c", MCP_MOCK_SERVER_SCRIPT),
        workspace_root=tmp_path,
    )

    client = MCPClient(config=config)
    await client.connect()

    try:
        # Discover tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "echo" in tool_names
        assert "file_reader" in tool_names

        # Call echo tool
        ctx = ToolContext(
            agent_id="a1",
            session_id="s1",
            workspace_root=tmp_path,
        )
        echo_res = await client.call_tool("echo", {"text": "hello mcp"}, ctx)
        assert echo_res.success is True
        assert echo_res.output == [{"type": "text", "text": "hello mcp"}]
        assert echo_res.isolation_level is IsolationLevel.WORKSPACE
        assert echo_res.provenance is not None
        assert echo_res.provenance.requested.provider == "mcp.test_server"
        assert echo_res.provenance.requested.model == "echo"

        # Call file_reader tool within workspace
        file_res = await client.call_tool("file_reader", {"file_path": str(test_file)}, ctx)
        assert file_res.success is True
        assert file_res.output == [{"type": "text", "text": "MCP workspace content"}]

        # Invoke tool via MCPTool instance
        echo_tool = next(t for t in tools if t.name == "echo")
        assert isinstance(echo_tool, MCPTool)
        mcp_tool_res = await echo_tool.execute({"text": "via MCPTool"}, ctx)
        assert mcp_tool_res.success is True
        assert mcp_tool_res.output == [{"type": "text", "text": "via MCPTool"}]

        # Call error_trigger tool
        err_res = await client.call_tool("error_trigger", {}, ctx)
        assert err_res.success is False
        assert "Custom internal error" in str(err_res.error)

    finally:
        await client.disconnect()


# ======================================================================================
# 4. Workspace Boundary and Path Traversal Enforcement
# ======================================================================================


@pytest.mark.asyncio
async def test_mcp_client_path_traversal_in_arguments_is_blocked(tmp_path: Path) -> None:
    """Tool invocation arguments escaping workspace_root raise PathTraversalError."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret outside")

    config = MCPConnectionConfig(
        server_name="test_server",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-c", MCP_MOCK_SERVER_SCRIPT),
        workspace_root=workspace,
    )

    async with MCPClient(config=config) as client:
        ctx = ToolContext(
            agent_id="a1",
            session_id="s1",
            workspace_root=workspace,
        )

        # Relative traversal
        with pytest.raises(PathTraversalError):
            await client.call_tool("file_reader", {"file_path": "../secret.txt"}, ctx)

        # Absolute outside path
        with pytest.raises(PathTraversalError):
            await client.call_tool("file_reader", {"file_path": str(outside_file)}, ctx)

        # Deeply nested traversal in dictionary argument
        with pytest.raises(PathTraversalError):
            await client.call_tool(
                "custom",
                {"nested": {"target": "../../outside"}},
                ctx,
            )


@pytest.mark.asyncio
async def test_mcp_client_command_and_arg_traversals_rejected(tmp_path: Path) -> None:
    """MCP server command or args containing escaping paths are rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Command escaping workspace root
    bad_cmd_cfg = MCPConnectionConfig(
        server_name="bad_cmd",
        transport=MCPTransport.STDIO,
        command="../escape.sh",
        workspace_root=workspace,
    )
    client_bad_cmd = MCPClient(config=bad_cmd_cfg)
    with pytest.raises(PathTraversalError):
        await client_bad_cmd.connect()

    # Arg escaping workspace root
    bad_arg_cfg = MCPConnectionConfig(
        server_name="bad_arg",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-f", "../outside.py"),
        workspace_root=workspace,
    )
    client_bad_arg = MCPClient(config=bad_arg_cfg)
    with pytest.raises(PathTraversalError):
        await client_bad_arg.connect()

    # Missing command on STDIO transport
    no_cmd_cfg = MCPConnectionConfig(
        server_name="no_cmd",
        transport=MCPTransport.STDIO,
        command=None,
        workspace_root=workspace,
    )
    client_no_cmd = MCPClient(config=no_cmd_cfg)
    with pytest.raises(ValueError, match="requires 'command' for STDIO transport"):
        await client_no_cmd.connect()


# ======================================================================================
# 5. Network Egress Policy Enforcement Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_mcp_network_egress_enforcement() -> None:
    """Remote transports (SSE/WEBSOCKET) enforce allow_network and egress_allowlist."""
    # Disallowed network
    cfg_no_net = MCPConnectionConfig(
        server_name="remote_mcp",
        transport=MCPTransport.SSE,
        url="https://api.example.com/mcp",
        isolation=NoIsolation(),
        allow_network=False,
    )
    client_no_net = MCPClient(config=cfg_no_net)
    with pytest.raises(SandboxViolationError, match="allow_network=False"):
        await client_no_net.connect()

    # Missing URL
    cfg_no_url = MCPConnectionConfig(
        server_name="remote_mcp",
        transport=MCPTransport.SSE,
        url=None,
        isolation=NoIsolation(),
        allow_network=True,
    )
    client_no_url = MCPClient(config=cfg_no_url)
    with pytest.raises(ValueError, match="requires a 'url'"):
        await client_no_url.connect()

    # Egress forbidden by egress_allowlist
    cfg_blocked_egress = MCPConnectionConfig(
        server_name="remote_mcp",
        transport=MCPTransport.WEBSOCKET,
        url="wss://unauthorized.domain.com/ws",
        isolation=NoIsolation(),
        allow_network=True,
        egress_allowlist=("api.github.com", "*.trusted.org"),
    )
    client_blocked = MCPClient(config=cfg_blocked_egress)
    with pytest.raises(SandboxViolationError, match="forbidden by egress_allowlist"):
        await client_blocked.connect()

    # Valid egress allowed (passes network policy check, then fails on unimplemented transport in call_tool)
    cfg_allowed_egress = MCPConnectionConfig(
        server_name="remote_mcp",
        transport=MCPTransport.WEBSOCKET,
        url="wss://sub.trusted.org/ws",
        isolation=NoIsolation(),
        allow_network=True,
        egress_allowlist=("api.github.com", "*.trusted.org"),
    )
    client_allowed = MCPClient(config=cfg_allowed_egress)
    await client_allowed.connect()
    assert client_allowed.is_connected is True

    ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=Path("/tmp"))
    res = await client_allowed.call_tool("remote_op", {"p": 1}, ctx)
    assert res.success is False
    assert "MCP transport 'websocket' is not implemented" in str(res.error)
    assert res.output is None
    await client_allowed.disconnect()


# ======================================================================================
# 6. Timeout and Process Lifecycle Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_mcp_client_timeout_handling(tmp_path: Path) -> None:
    """Tool calls exceeding context.timeout_seconds gracefully return timeout failure."""
    # Server script that sleeps on tools/call
    hang_script = """
import sys, json, time

while True:
    line = sys.stdin.readline()
    if not line:
        break
    req = json.loads(line)
    if req.get("method") == "initialize":
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": {"protocolVersion": "2024-11-05"}}) + "\\n")
        sys.stdout.flush()
    elif req.get("method") == "tools/call":
        time.sleep(5)
"""
    config = MCPConnectionConfig(
        server_name="slow_server",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-c", hang_script),
        workspace_root=tmp_path,
    )

    async with MCPClient(config=config) as client:
        ctx = ToolContext(
            agent_id="a1",
            session_id="s1",
            workspace_root=tmp_path,
            timeout_seconds=0.2,
        )
        res = await client.call_tool("slow_tool", {}, ctx)
        assert res.success is False
        assert "timed out after 0.2s" in str(res.error)


@pytest.mark.asyncio
async def test_mcp_client_executable_not_found(tmp_path: Path) -> None:
    """Non-existent executable raises FileNotFoundError on connect."""
    config = MCPConnectionConfig(
        server_name="missing_server",
        transport=MCPTransport.STDIO,
        command="non_existent_mcp_binary_xyz_123",
        workspace_root=tmp_path,
    )
    client = MCPClient(config=config)
    with pytest.raises(FileNotFoundError, match="non_existent_mcp_binary_xyz_123"):
        await client.connect()


@pytest.mark.asyncio
async def test_mcp_client_non_stdio_transport_fails_explicitly(tmp_path: Path) -> None:
    """An unimplemented transport fails explicitly with a descriptive error (P6 / Issue #32).

    Streamable HTTP (`http`) is implemented and tested in `test_tools_mcp_http.py`; legacy
    SSE and WebSocket remain unimplemented, and must still say so rather than return nothing.
    """
    for transport in (MCPTransport.SSE, MCPTransport.WEBSOCKET):
        config = MCPConnectionConfig(
            server_name=f"remote_{transport.value}",
            transport=transport,
            url="https://api.github.com/mcp/sse"
            if transport == MCPTransport.SSE
            else "wss://api.github.com/mcp/ws",
            allow_network=True,
            isolation=NoIsolation(),
        )
        client = MCPClient(config=config)
        await client.connect()
        assert client.is_connected is True

        with pytest.raises(
            NotImplementedError, match=f"MCP transport '{transport.value}' is not implemented"
        ):
            await client.list_tools()

        ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
        res = await client.call_tool("remote_op", {"arg": 123}, ctx)
        assert res.success is False
        assert res.output is None
        assert f"MCP transport '{transport.value}' is not implemented" in str(res.error)
        assert res.provenance is not None
        assert res.provenance.requested.provider == f"mcp.remote_{transport.value}"
        assert res.provenance.requested.model == "remote_op"
        await client.disconnect()


def test_mcp_config_remote_transport_refuses_filesystem_isolation(tmp_path: Path) -> None:
    """A remote MCP transport (SSE / Streamable HTTP / WebSocket) refuses filesystem isolation policies at construction (Issue #64)."""
    for transport in (MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP, MCPTransport.WEBSOCKET):
        # Default WorkspaceIsolation must be rejected
        with pytest.raises(ValueError, match="cannot accept filesystem-shaped isolation policy"):
            MCPConnectionConfig(
                server_name="remote_err",
                transport=transport,
                url="https://example.com",
            )

        # Explicit ContainerIsolation must be rejected
        with pytest.raises(ValueError, match="cannot accept filesystem-shaped isolation policy"):
            MCPConnectionConfig(
                server_name="remote_err",
                transport=transport,
                url="https://example.com",
                isolation=ContainerIsolation(image="alpine", allow_network=True),
                allow_network=True,
            )

        # Explicit NoIsolation but workspace_root provided must be rejected
        with pytest.raises(ValueError, match="cannot accept 'workspace_root'"):
            MCPConnectionConfig(
                server_name="remote_err",
                transport=transport,
                url="https://example.com",
                isolation=NoIsolation(),
                workspace_root=tmp_path,
            )


@pytest.mark.asyncio
async def test_local_tool_without_handler_raises(tmp_path: Path) -> None:
    """LocalTool with no handler raises RuntimeError rather than fabricating success (P6 / Issue #43)."""
    tool = LocalTool(name="unhandled_tool", description="A tool with no execution handler")
    ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=tmp_path)
    with pytest.raises(RuntimeError, match="has no execution handler configured"):
        await tool.execute({}, ctx)


def test_mcp_connection_config_omitted_workspace_root() -> None:
    """
    Test that omitted workspace_root with STDIO transport defaults correctly to Path.cwd()
    and maintains WorkspaceIsolation.

    Killed by: src/uclone_x/tools/models.py :: iso is None
    Becomes: iso is not None
    """
    from pathlib import Path

    from uclone_x.sandbox.models import IsolationLevel
    from uclone_x.tools.models import MCPConnectionConfig, MCPTransport

    # 1. Test omission
    cfg = MCPConnectionConfig(server_name="test_omission", command="/bin/true")
    assert cfg.transport is MCPTransport.STDIO
    assert cfg.isolation.level is IsolationLevel.WORKSPACE
    assert cfg.workspace_root == Path.cwd()

    # 2. Test explicit None doesn't break if isolation defaults to workspace
    cfg_none = MCPConnectionConfig(
        server_name="test_explicit_none", command="/bin/true", workspace_root=None
    )
    assert cfg_none.isolation.level is IsolationLevel.WORKSPACE
    assert cfg_none.workspace_root == Path.cwd()


@pytest.mark.asyncio
async def test_mcp_client_info_reports_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP `initialize` handshake announces the live package version.

    `clientInfo.version` is what every MCP server this client connects to records
    about us, and #1131 found it two releases behind. Asserting agreement with
    `uclone_x.__version__` would pass against a literal that happens to be correct
    today -- the defect itself. So this moves `__version__` to a value no literal in
    the tree carries and reads the request the server actually received, rather than
    the dict the client meant to build.

    Killed by: src/uclone_x/tools/client.py :: "version": uclone_x.__version__,
    Becomes: "version": "0.0.0",
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    capture = tmp_path / "initialize_request.json"
    recorder_script = f"""
import sys, json
line = sys.stdin.readline()
req = json.loads(line)
open({str(capture)!r}, "w").write(line)
sys.stdout.write(
    json.dumps({{"jsonrpc": "2.0", "id": req.get("id"), "result": {{}}}}) + "\\n"
)
sys.stdout.flush()
sys.stdin.readline()
"""
    config = MCPConnectionConfig(
        server_name="version_recorder",
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=("-c", recorder_script),
        workspace_root=tmp_path,
    )
    client = MCPClient(config=config)
    await client.connect()
    try:
        assert client.is_connected is True
    finally:
        await client.disconnect()

    recorded: dict[str, Any] = json.loads(capture.read_text(encoding="utf-8"))
    client_info = recorded["params"]["clientInfo"]
    assert client_info["name"] == "uclone-x"
    assert client_info["version"] == "9.8.7-probe"
