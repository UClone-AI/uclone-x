"""Dynamic Model Context Protocol (MCP) configuration loader and tool registry auto-wiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from uclone_x.sandbox.models import (
    ContainerIsolation,
    IsolationPolicy,
    NoIsolation,
    WasmIsolation,
    WorkspaceIsolation,
)
from uclone_x.tools.client import MCPClient
from uclone_x.tools.models import MCPConnectionConfig, MCPTransport
from uclone_x.tools.protocols import MCPClientProtocol, ToolProtocol, ToolRegistryProtocol

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}|\$([A-Za-z0-9_]+)")

__all__ = [
    "MCPConfigFileLoader",
    "expand_env_vars",
]


def expand_env_vars(value: str) -> str:
    """Expand ${VAR}, ${VAR:-default}, and $VAR in a string using os.environ."""

    def _repl(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(3)
        default_val = match.group(2)
        if var_name in os.environ:
            val = os.environ[var_name]
            if val != "" or default_val is None:
                return val
        if default_val is not None:
            return default_val
        return ""

    return _ENV_VAR_PATTERN.sub(_repl, value)


class MCPConfigFileLoader:
    """Discovers, parses, and instantiates MCP servers from standard configuration files."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._config_path = config_path

    @property
    def workspace_root(self) -> Path | None:
        """Workspace root directory for path isolation."""
        return self._workspace_root

    @property
    def config_path(self) -> Path | None:
        """Explicitly configured configuration file path."""
        return self._config_path

    def discover_config_file(self) -> Path | None:
        """Discover an MCP config file at explicit config_path or default candidate locations."""
        if self._config_path is not None:
            p = self._config_path.expanduser()
            return p if p.is_file() else None

        candidates: list[Path] = []
        if self._workspace_root is not None:
            candidates.append(self._workspace_root / ".uclone" / "mcp.json")
            candidates.append(self._workspace_root / "mcp.json")

        cwd = Path.cwd()
        candidates.append(cwd / ".uclone" / "mcp.json")
        candidates.append(cwd / "mcp.json")

        home = Path.home()
        candidates.append(home / ".uclone" / "mcp.json")
        candidates.append(home / ".config" / "uclone" / "mcp.json")

        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @classmethod
    def substitute_env(cls, obj: Any) -> Any:
        """Recursively substitute environment variables in dicts, lists, and strings."""
        if isinstance(obj, str):
            return expand_env_vars(obj)
        if isinstance(obj, list):
            items = cast(list[object], obj)
            return [cls.substitute_env(item) for item in items]
        if isinstance(obj, dict):
            raw_dict = cast(dict[object, object], obj)
            return {str(k): cls.substitute_env(v) for k, v in raw_dict.items()}
        return obj

    def _build_server_config(
        self, server_name: str, raw_cfg: dict[str, Any]
    ) -> MCPConnectionConfig:
        """Build and validate an MCPConnectionConfig instance from a raw server configuration dictionary."""
        # `type` is the key Claude Desktop / Claude Code snippets use for the same field, and
        # those snippets are what MCP vendors publish for users to paste.
        transport_raw = raw_cfg.get("transport", raw_cfg.get("type"))
        command_raw = raw_cfg.get("command")
        url_raw = raw_cfg.get("url")
        args_raw = raw_cfg.get("args", [])
        env_raw = raw_cfg.get("env", {})
        headers_raw = raw_cfg.get("headers", {})
        env_allowlist_raw = raw_cfg.get(
            "env_allowlist", raw_cfg.get("envAllowlist", raw_cfg.get("env_allow_list", []))
        )
        allow_network_raw = raw_cfg.get("allow_network", raw_cfg.get("allowNetwork", None))
        egress_allowlist_raw = raw_cfg.get(
            "egress_allowlist",
            raw_cfg.get("egressAllowlist", raw_cfg.get("egress_allow_list", [])),
        )
        isolation_raw = raw_cfg.get("isolation")
        workspace_root_raw = raw_cfg.get("workspace_root", raw_cfg.get("workspaceRoot"))
        image_raw = raw_cfg.get("image", "alpine:latest")

        # 1. Transport resolution
        # When transport is omitted:
        # - If `url` is present and `command` is absent, transport is inferred as SSE (MCPTransport.SSE).
        #   Note: Streamable-HTTP servers (MCPTransport.STREAMABLE_HTTP) with only a url must explicitly specify
        #   transport="http" (or "streamable_http") to avoid being inferred as SSE.
        # - Otherwise, transport defaults to local STDIO (MCPTransport.STDIO).
        transport: MCPTransport
        if transport_raw is not None:
            raw_str = str(transport_raw).lower().replace("-", "_")
            if raw_str in ("streamable_http", "streamablehttp"):
                raw_str = MCPTransport.STREAMABLE_HTTP.value
            try:
                transport = MCPTransport(raw_str)
            except ValueError as err:
                raise ValueError(
                    f"Unknown MCP transport '{transport_raw}' for server '{server_name}'. "
                    f"Valid transports are: {[t.value for t in MCPTransport]}"
                ) from err
        elif url_raw is not None and command_raw is None:
            # Inference rule: A descriptor providing a `url` without a `command` defaults to SSE
            # (the historical MCP remote default). Streamable HTTP servers must explicitly set
            # transport="http" or transport="streamable_http".
            transport = MCPTransport.SSE
        else:
            transport = MCPTransport.STDIO

        # 2. Command, URL, and Args
        command: str | None = str(command_raw) if command_raw is not None else None
        url: str | None = str(url_raw) if url_raw is not None else None

        args: tuple[str, ...]
        if isinstance(args_raw, (list, tuple)):
            args = tuple(str(a) for a in cast(Sequence[object], args_raw))
        else:
            args = ()

        # 3. Environment & Allowlist
        env: dict[str, str] = {}
        if isinstance(env_raw, dict):
            raw_env_map = cast(dict[object, object], env_raw)
            env = {str(k): str(v) for k, v in raw_env_map.items()}

        env_allowlist: tuple[str, ...]
        if isinstance(env_allowlist_raw, (list, tuple)):
            env_allowlist = tuple(str(e) for e in cast(Sequence[object], env_allowlist_raw))
        else:
            env_allowlist = ()

        headers: dict[str, str] = {}
        if isinstance(headers_raw, dict):
            raw_headers_map = cast(dict[object, object], headers_raw)
            headers = {str(k): str(v) for k, v in raw_headers_map.items()}

        # 4. Workspace Root Resolution
        ws_root: Path | None = None
        if workspace_root_raw is not None:
            ws_root = Path(str(workspace_root_raw)).expanduser().resolve()
        elif self._workspace_root is not None:
            ws_root = self._workspace_root.resolve()
        elif transport not in (MCPTransport.SSE, MCPTransport.WEBSOCKET):
            ws_root = Path.cwd()

        # 5. Network Policy Resolution
        allow_network: bool
        if allow_network_raw is not None:
            allow_network = bool(allow_network_raw)
        else:
            allow_network = transport in (
                MCPTransport.SSE,
                MCPTransport.STREAMABLE_HTTP,
                MCPTransport.WEBSOCKET,
            )

        egress_allowlist: tuple[str, ...]
        if isinstance(egress_allowlist_raw, (list, tuple)):
            egress_allowlist = tuple(str(e) for e in cast(Sequence[object], egress_allowlist_raw))
        else:
            egress_allowlist = ()

        # 6. Isolation Policy Resolution
        isolation: IsolationPolicy
        if isolation_raw is not None:
            if isinstance(isolation_raw, str):
                iso_str = isolation_raw.lower().strip()
                if iso_str == "none":
                    isolation = NoIsolation()
                elif iso_str == "workspace":
                    isolation = WorkspaceIsolation()
                elif iso_str == "container":
                    isolation = ContainerIsolation(
                        image=str(image_raw),
                        allow_network=allow_network,
                        egress_allowlist=egress_allowlist,
                    )
                elif iso_str == "wasm":
                    isolation = WasmIsolation(allow_network=allow_network)
                else:
                    raise ValueError(
                        f"Unsupported isolation type '{isolation_raw}' for server '{server_name}'"
                    )
            elif isinstance(
                isolation_raw, (NoIsolation, WorkspaceIsolation, ContainerIsolation, WasmIsolation)
            ):
                isolation = isolation_raw
            else:
                raise ValueError(f"Invalid isolation specification for server '{server_name}'")
        else:
            if transport in (
                MCPTransport.SSE,
                MCPTransport.STREAMABLE_HTTP,
                MCPTransport.WEBSOCKET,
            ):
                isolation = NoIsolation()
            else:
                isolation = WorkspaceIsolation()

        # Remote transports require NoIsolation and no workspace_root
        if transport in (
            MCPTransport.SSE,
            MCPTransport.STREAMABLE_HTTP,
            MCPTransport.WEBSOCKET,
        ):
            isolation = NoIsolation()
            ws_root = None

        return MCPConnectionConfig(
            server_name=server_name,
            transport=transport,
            command=command,
            args=args,
            url=url,
            headers=headers,
            env=env,
            env_allowlist=env_allowlist,
            workspace_root=ws_root,
            isolation=isolation,
            allow_network=allow_network,
            egress_allowlist=egress_allowlist,
        )

    def parse_config_dict(
        self, data: Any, source_path: Path | None = None
    ) -> list[MCPConnectionConfig]:
        """Parse MCP connection configurations from raw dictionary or list data."""
        expanded: Any = self.substitute_env(data)
        configs: list[MCPConnectionConfig] = []

        if isinstance(expanded, list):
            items = cast(list[object], expanded)
            for idx, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                item_dict = cast(dict[str, Any], item)
                name = str(
                    item_dict.get("server_name") or item_dict.get("name") or f"mcp_server_{idx}"
                )
                configs.append(self._build_server_config(name, item_dict))
            return configs

        if isinstance(expanded, dict):
            raw_dict = cast(dict[str, Any], expanded)
            servers_map: dict[str, Any]
            if "mcpServers" in raw_dict and isinstance(raw_dict["mcpServers"], dict):
                servers_map = cast(dict[str, Any], raw_dict["mcpServers"])
            elif "servers" in raw_dict and isinstance(raw_dict["servers"], dict):
                servers_map = cast(dict[str, Any], raw_dict["servers"])
            elif all(isinstance(v, dict) for v in raw_dict.values()):
                servers_map = raw_dict
            else:
                raise ValueError(
                    "MCP config dictionary must contain 'mcpServers' mapping or direct server definitions"
                )

            for name, s_data in servers_map.items():
                if not isinstance(s_data, dict):
                    continue
                s_dict = cast(dict[str, Any], s_data)
                configs.append(self._build_server_config(str(name), s_dict))
            return configs

        raise ValueError(
            f"Invalid MCP configuration format{' in ' + str(source_path) if source_path else ''}: expected dict or list"
        )

    def parse_config_json(
        self, json_str: str, source_path: Path | None = None
    ) -> list[MCPConnectionConfig]:
        """Parse and validate MCP server configurations from JSON string."""
        try:
            raw_data = cast(object, json.loads(json_str))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in MCP configuration: {exc}") from exc

        return self.parse_config_dict(raw_data, source_path=source_path)

    def load_from_file(self, file_path: Path | str) -> list[MCPConnectionConfig]:
        """Parse MCP server configurations from a specific file path."""
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MCP configuration file not found: '{path}'")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to read MCP config file '{path}': {exc}") from exc

        return self.parse_config_json(content, source_path=path)

    def load_configs(self) -> list[MCPConnectionConfig]:
        """Discover and parse MCP server configurations."""
        target_path = self.discover_config_file()
        if target_path is None:
            return []
        return self.load_from_file(target_path)

    def create_clients(
        self, configs: Sequence[MCPConnectionConfig] | None = None
    ) -> list[MCPClient]:
        """Instantiate MCPClient instances for the given or discovered configurations."""
        cfgs = configs if configs is not None else self.load_configs()
        return [MCPClient(config=cfg) for cfg in cfgs]

    async def register_client_tools_async(
        self,
        registry: ToolRegistryProtocol,
        client: MCPClientProtocol,
    ) -> list[ToolProtocol]:
        """Discover tools from an MCPClient and register them into registry."""
        server_name = getattr(client.config, "server_name", "unknown")
        try:
            tools = await client.list_tools()
            for tool in tools:
                registry.register(tool)
            logger.info("Registered %d tools from MCP server '%s'", len(tools), server_name)
            return tools
        except Exception as exc:
            logger.warning(
                "Skipping MCP server '%s' due to discovery failure: %s",
                server_name,
                exc,
            )
            return []

    def register_client_tools(
        self,
        registry: ToolRegistryProtocol,
        client: MCPClientProtocol,
    ) -> list[ToolProtocol]:
        """Synchronously discover and register tools from an MCPClient."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            return asyncio.run(self._register_on_temporary_loop(registry, client))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self._register_on_temporary_loop(registry, client)
            )
            return future.result(timeout=30.0)

    async def _register_on_temporary_loop(
        self,
        registry: ToolRegistryProtocol,
        client: MCPClientProtocol,
    ) -> list[ToolProtocol]:
        """List and register a client's tools, then disconnect it before the loop closes.

        The sync entry points run in a loop of their own that closes on return (#1414). A
        stdio server's pipes belong to the loop that started it, so a client left connected
        here would answer every later call with "attached to a different loop". Stopping
        the server now leaves the client disconnected, and `call_tool` starts it again from
        whichever loop first uses one of its tools.
        """
        try:
            return await self.register_client_tools_async(registry, client)
        finally:
            try:
                await client.disconnect()
            except Exception as exc:
                server_name = getattr(client.config, "server_name", "unknown")
                logger.warning("Could not stop MCP server '%s' after listing: %s", server_name, exc)

    async def _load_and_register_on_temporary_loop(
        self,
        registry: ToolRegistryProtocol,
        cfgs: Sequence[MCPConnectionConfig],
    ) -> list[ToolProtocol]:
        all_registered: list[ToolProtocol] = []
        for cfg in cfgs:
            client = MCPClient(config=cfg)
            all_registered.extend(await self._register_on_temporary_loop(registry, client))
        return all_registered

    async def load_and_register_tools_async(
        self,
        registry: ToolRegistryProtocol,
        configs: Sequence[MCPConnectionConfig] | None = None,
    ) -> list[ToolProtocol]:
        """Asynchronously discover and register tools from all configured MCP servers."""
        cfgs = configs if configs is not None else self.load_configs()
        all_registered: list[ToolProtocol] = []

        for cfg in cfgs:
            client = MCPClient(config=cfg)
            registered = await self.register_client_tools_async(registry, client)
            all_registered.extend(registered)

        return all_registered

    def load_and_register_tools(
        self,
        registry: ToolRegistryProtocol,
        configs: Sequence[MCPConnectionConfig] | None = None,
    ) -> list[ToolProtocol]:
        """Synchronously discover and register tools from all configured MCP servers."""
        cfgs = configs if configs is not None else self.load_configs()
        if not cfgs:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            return asyncio.run(self._load_and_register_on_temporary_loop(registry, cfgs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self._load_and_register_on_temporary_loop(registry, cfgs)
            )
            return future.result(timeout=30.0)
