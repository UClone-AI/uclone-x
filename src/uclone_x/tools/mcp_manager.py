"""External MCP servers a user connects from Settings: stored, connected, and registered.

A server added here lives in `<storage_dir>/mcp_servers.json`, in the same `mcpServers`
shape Claude Desktop and Claude Code write, so a snippet copied from a server's own README
can be pasted in as it is. Each connected server's tools are registered as
`<server>__<tool>`, so two servers that both offer `search` cannot shadow each other or a
built-in tool; a name that would still collide is reported, never overwritten.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from uclone_x.sandbox.models import NoIsolation
from uclone_x.tools.client import MCPClient
from uclone_x.tools.models import MCPConnectionConfig, MCPTransport
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol

SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

#: Host variables a local server is given. Deny-by-default still holds for everything else:
#: without PATH and HOME, `npx` and `uvx` -- how nearly every published server is launched --
#: cannot even be found. None of these is credential-shaped (`is_secret_env_name`).
LOCAL_SERVER_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SHELL",
)

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")

Status = Literal["connecting", "connected", "error", "disabled"]


class MCPServerSpec(BaseModel):
    """One server as the user configured it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    transport: Literal["http", "stdio"]
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _name_shape(cls, value: str) -> str:
        if not SERVER_NAME_PATTERN.match(value):
            raise ValueError(
                "A server name may use letters, digits, '-' and '_', up to 32 characters"
            )
        return value

    @field_validator("headers", "env", mode="before")
    @classmethod
    def _trim_values(cls, value: object) -> object:
        # A token pasted from a web page often carries a trailing newline. Trimmed here,
        # it would otherwise make the request fail with the value quoted in the error.
        if isinstance(value, dict):
            items = cast(dict[object, object], value).items()
            return {k: v.strip() if isinstance(v, str) else v for k, v in items}
        return value

    @model_validator(mode="after")
    def _transport_shape(self) -> MCPServerSpec:
        # The message names the field, never the value: a header value is a credential.
        for label, texts in (
            ("address", [self.url or ""]),
            ("header", [*self.headers, *self.headers.values()]),
            ("command", [self.command or "", *self.args]),
            ("environment variable", [*self.env, *self.env.values()]),
        ):
            if any(_CONTROL_CHARACTERS.search(text) for text in texts):
                raise ValueError(f"A {label} contains a line break or control character")
        if self.transport == "http":
            if not self.url or not self.url.startswith(("http://", "https://")):
                raise ValueError("A remote server needs an address starting with https://")
            if self.command or self.args or self.env:
                raise ValueError("A remote server takes an address, not a command")
        else:
            if not self.command or not self.command.strip():
                raise ValueError("A local server needs the command that starts it")
            if self.url or self.headers:
                raise ValueError("A local server takes a command, not an address")
        return self

    def to_file_entry(self) -> dict[str, Any]:
        entry: dict[str, Any]
        if self.transport == "http":
            entry = {"type": "http", "url": self.url}
            if self.headers:
                entry["headers"] = dict(self.headers)
        else:
            entry = {"command": self.command, "args": list(self.args)}
            if self.env:
                entry["env"] = dict(self.env)
        if not self.enabled:
            entry["disabled"] = True
        return entry


def spec_from_entry(name: str, raw: object) -> MCPServerSpec:
    """Read one `mcpServers` entry: Claude Desktop, Claude Code and our own file alike.

    Raises ValueError naming what is wrong with the entry.
    """
    if not isinstance(raw, dict):
        raise ValueError("the entry is not an object")
    entry = {str(k): v for k, v in cast(dict[object, object], raw).items()}
    kind = entry.get("type", entry.get("transport"))
    url = entry.get("url")
    if isinstance(kind, str):
        kind = kind.lower().replace("-", "_")
    if kind in ("http", "streamable_http", "streamablehttp") or (kind is None and url):
        transport: Literal["http", "stdio"] = "http"
    elif kind in (None, "stdio"):
        transport = "stdio"
    else:
        raise ValueError(
            f"its transport '{kind}' is not supported; use a Streamable HTTP address or a command"
        )
    try:
        return MCPServerSpec(
            name=name,
            transport=transport,
            url=url if isinstance(url, str) else None,
            headers=_str_map(entry.get("headers")),
            command=cast(str, entry["command"]) if isinstance(entry.get("command"), str) else None,
            args=tuple(str(a) for a in cast(list[object], entry.get("args") or [])),
            env=_str_map(entry.get("env")),
            enabled=not bool(entry.get("disabled", False)),
        )
    except ValidationError as err:
        raise ValueError(_first_error(err)) from err


def _str_map(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'headers' and 'env' must be objects of text values")
    return {str(k): str(v) for k, v in cast(dict[object, object], raw).items()}


def _first_error(err: ValidationError) -> str:
    first = err.errors()[0]
    msg = str(first.get("msg", err))
    return msg.removeprefix("Value error, ")


@dataclass
class _Server:
    spec: MCPServerSpec
    status: Status = "connecting"
    error: str | None = None
    client: MCPClient | None = None
    tools: list[ToolProtocol] = field(default_factory=lambda: [])
    registered: list[str] = field(default_factory=lambda: [])


ClientFactory = Callable[[MCPConnectionConfig, str], MCPClient]


class DuplicateServerError(ValueError):
    """A server with this name already exists."""


class UnknownServerError(KeyError):
    """No server has this name."""


class MCPServerManager:
    """Owns the user's external MCP servers for one running app (one event loop)."""

    def __init__(
        self,
        registry: ToolRegistryProtocol,
        config_path: Path,
        workspace_root: Path,
        client_factory: ClientFactory | None = None,
        connect_timeout: float = 20.0,
    ) -> None:
        self._registry = registry
        self._config_path = config_path
        self._workspace_root = workspace_root
        self._connect_timeout = connect_timeout
        self._client_factory: ClientFactory = client_factory or (
            lambda config, prefix: MCPClient(
                config=config, tool_name_prefix=prefix, connect_timeout=connect_timeout
            )
        )
        self._servers: dict[str, _Server] = {}
        self._lock = asyncio.Lock()
        self._load_error: str | None = None
        self._load()

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def load_error(self) -> str | None:
        """Why the stored file could not be read, if it could not."""
        return self._load_error

    # ── storage ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._config_path.exists():
            return
        try:
            data = cast(object, json.loads(self._config_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as err:
            self._load_error = f"{self._config_path} could not be read: {err}"
            return
        raw_servers = (
            cast(dict[str, object], data).get("mcpServers") if isinstance(data, dict) else None
        )
        if not isinstance(raw_servers, dict):
            return
        for name, raw in cast(dict[str, object], raw_servers).items():
            try:
                spec = spec_from_entry(name, raw)
            except ValueError as err:
                # Kept visible, not dropped: the user wrote it and should see why it is idle.
                self._load_error = f"Server '{name}' in {self._config_path} is invalid: {err}"
                continue
            self._servers[name] = _Server(
                spec=spec, status="connecting" if spec.enabled else "disabled"
            )

    def _save(self) -> None:
        payload = {
            "mcpServers": {name: s.spec.to_file_entry() for name, s in self._servers.items()}
        }
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        # Header and env values are credentials: owner-only, written atomically so a crash
        # mid-write cannot leave a half file that loses every server.
        fd, tmp = tempfile.mkstemp(dir=self._config_path.parent, prefix=".mcp_servers.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._config_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── connection ───────────────────────────────────────────────────────

    def _connection_config(self, spec: MCPServerSpec) -> MCPConnectionConfig:
        if spec.transport == "http":
            return MCPConnectionConfig(
                server_name=spec.name,
                transport=MCPTransport.STREAMABLE_HTTP,
                url=spec.url,
                headers=dict(spec.headers),
                allow_network=True,
                isolation=NoIsolation(),
            )
        return MCPConnectionConfig(
            server_name=spec.name,
            transport=MCPTransport.STDIO,
            command=spec.command,
            args=spec.args,
            env=dict(spec.env),
            env_allowlist=LOCAL_SERVER_ENV_ALLOWLIST,
            workspace_root=self._workspace_root,
        )

    async def _connect(self, server: _Server) -> None:
        server.status = "connecting"
        server.error = None
        client = self._client_factory(self._connection_config(server.spec), server.spec.name)
        server.client = client
        try:
            await asyncio.wait_for(client.connect(), timeout=self._connect_timeout)
            tools = await asyncio.wait_for(client.list_tools(), timeout=self._connect_timeout)
        except Exception as exc:  # every failure is shown to the user, with its cause
            await self._drop_client(server)
            server.status = "error"
            server.error = _connect_error_text(server.spec, exc)
            return
        server.tools = tools
        skipped: list[str] = []
        for tool in tools:
            if self._registry.get(tool.name) is not None:
                skipped.append(tool.name)
                continue
            self._registry.register(tool)
            server.registered.append(tool.name)
        server.status = "connected"
        if skipped:
            server.error = (
                "Some tools were not added because another tool already has the same name: "
                + ", ".join(skipped)
            )

    async def _drop_client(self, server: _Server) -> None:
        for name in server.registered:
            self._registry.unregister(name)
        server.registered = []
        server.tools = []
        client, server.client = server.client, None
        if client is not None:
            try:
                await client.retire()
            except Exception:
                pass

    async def start(self) -> None:
        """Connect every enabled stored server, concurrently. Failures stay per server."""
        async with self._lock:
            pending = [s for s in self._servers.values() if s.spec.enabled]
            await asyncio.gather(*(self._connect(s) for s in pending))

    async def close(self) -> None:
        async with self._lock:
            for server in self._servers.values():
                await self._drop_client(server)

    # ── user actions ─────────────────────────────────────────────────────

    async def add(self, spec: MCPServerSpec) -> dict[str, Any]:
        """Store the server, then try to connect. Saved even when the connection fails."""
        async with self._lock:
            if spec.name in self._servers:
                raise DuplicateServerError(f"A server named '{spec.name}' already exists")
            server = _Server(spec=spec, status="connecting" if spec.enabled else "disabled")
            self._servers[spec.name] = server
            self._save()
            if spec.enabled:
                await self._connect(server)
            return self._view(server)

    async def remove(self, name: str) -> None:
        async with self._lock:
            server = self._require(name)
            await self._drop_client(server)
            del self._servers[name]
            self._save()

    async def set_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        async with self._lock:
            server = self._require(name)
            server.spec = server.spec.model_copy(update={"enabled": enabled})
            self._save()
            await self._drop_client(server)
            if enabled:
                await self._connect(server)
            else:
                server.status = "disabled"
                server.error = None
            return self._view(server)

    async def reconnect(self, name: str) -> dict[str, Any]:
        async with self._lock:
            server = self._require(name)
            await self._drop_client(server)
            if server.spec.enabled:
                await self._connect(server)
            return self._view(server)

    async def import_json(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Add every server in a pasted `mcpServers` snippet. ValueError if it is not JSON.

        Accepts `{"mcpServers": {...}}`, a bare `{name: entry}` map, or a single entry
        wrapped by name -- the three shapes server READMEs publish.
        """
        try:
            data = cast(object, json.loads(text))
        except ValueError as err:
            raise ValueError(f"This is not valid JSON: {err}") from err
        if not isinstance(data, dict):
            raise ValueError('Expected an object such as {"mcpServers": {...}}')
        root = cast(dict[str, object], data)
        entries_obj = root.get("mcpServers", root.get("servers", root))
        if not isinstance(entries_obj, dict) or not entries_obj:
            raise ValueError("No servers were found in it")
        added: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for name, raw in cast(dict[str, object], entries_obj).items():
            try:
                spec = spec_from_entry(name, raw)
                added.append(await self.add(spec))
            except DuplicateServerError as err:
                skipped.append({"name": name, "reason": str(err)})
            except ValueError as err:
                skipped.append({"name": name, "reason": str(err)})
        return added, skipped

    # ── views ────────────────────────────────────────────────────────────

    def _require(self, name: str) -> _Server:
        server = self._servers.get(name)
        if server is None:
            raise UnknownServerError(name)
        return server

    def views(self) -> list[dict[str, Any]]:
        return [self._view(s) for s in self._servers.values()]

    def _view(self, server: _Server) -> dict[str, Any]:
        """What the dashboard may see. Header and env *values* never leave the Core."""
        spec = server.spec
        return {
            "name": spec.name,
            "transport": spec.transport,
            "url": _redact_query(spec.url),
            "command": spec.command,
            "args": list(spec.args),
            "env_keys": sorted(spec.env),
            "header_keys": sorted(spec.headers),
            "enabled": spec.enabled,
            "status": server.status,
            "error": server.error,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in server.tools
                if t.name in server.registered
            ],
        }


def _redact_query(url: str | None) -> str | None:
    """Hide query values: some servers take their key as `?api_key=...`."""
    if url is None:
        return None
    parts = urllib.parse.urlsplit(url)
    if not parts.query:
        return url
    keys = [k for k, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)]
    return parts._replace(query="&".join(f"{k}=..." for k in keys)).geturl()


def _connect_error_text(spec: MCPServerSpec, exc: BaseException) -> str:
    """Say why the connection failed and what would fix it (P6)."""
    if isinstance(exc, TimeoutError):
        if spec.transport == "stdio":
            return (
                "The server started but did not answer in time. Check that the command "
                "is right; a first run of npx or uvx may need longer to download."
            )
        return "The server did not answer in time. Check the address and your connection."
    if isinstance(exc, FileNotFoundError):
        return (
            f"The command '{spec.command}' was not found on this computer. "
            "Install it, or give its full path."
        )
    if isinstance(exc, httpx.ConnectError):
        return f"Could not reach {_redact_query(spec.url)}. Check the address and your connection."
    return str(exc) or type(exc).__name__
