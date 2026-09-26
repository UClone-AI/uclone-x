"""MCP (Model Context Protocol) client and tool integration with isolation boundaries."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shutil
import signal
import time
import types
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import httpx
from pydantic import JsonValue

import uclone_x
from uclone_x.core.provenance import Provenance
from uclone_x.errors import SandboxViolationError
from uclone_x.sandbox.models import (
    IsolationLevel,
    effective_isolation_level,
    is_secret_env_name,
)
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.sandbox.protocols import PathValidatorProtocol
from uclone_x.sandbox.story_jail import story_library_jail
from uclone_x.tools.models import (
    MCPConnectionConfig,
    MCPTransport,
    ToolContext,
    ToolResult,
)
from uclone_x.tools.protocols import MCPClientProtocol, ToolProtocol


def _command_exists(command: str, env: dict[str, str], cwd: str | None) -> bool:
    """Whether `command` names a program `sandbox-exec` can start with this `PATH` and cwd."""
    if os.sep in command or (os.altsep is not None and os.altsep in command):
        path = Path(command)
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(command, path=env.get("PATH", os.defpath)) is not None


class MCPTool:
    """A tool proxying execution to an MCP server."""

    #: Counts as a file-writing tool, so `enable_write_tools: false` refuses it (#1167).
    #: What an MCP server's tool does is the server's business: this proxy does not read
    #: the tool's `readOnlyHint` annotation, and MCP's own default for an unannotated tool
    #: is "not read-only". Refusing it is the only answer that cannot widen the flag.
    writes_files: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        client: MCPClientProtocol,
        remote_name: str | None = None,
    ) -> None:
        self._name = name
        #: What the server calls this tool, when the name shown to the model differs.
        self._remote_name = remote_name if remote_name is not None else name
        self._description = description
        self._parameters_schema = parameters_schema
        self._client = client

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        return self._name

    @property
    def description(self) -> str:
        """Detailed functional description for LLM reasoning."""
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining tool input arguments."""
        return self._parameters_schema

    async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool on the MCP server."""
        return await self._client.call_tool(self._remote_name, params, context)


class MCPClient:
    """Client for connecting to and invoking tools on Model Context Protocol servers.

    Enforces workspace isolation boundaries, network egress allowlists, and host
    environment variable scrubbing (deny-by-default for credentials).
    """

    def __init__(
        self,
        config: MCPConnectionConfig,
        validator: PathValidatorProtocol | None = None,
        tool_name_prefix: str | None = None,
        connect_timeout: float = 10.0,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._tool_name_prefix = tool_name_prefix
        self._connect_timeout = connect_timeout
        self._http_transport = http_transport
        self._validator: PathValidatorProtocol = (
            validator if validator is not None else PathValidator()
        )
        self._process: asyncio.subprocess.Process | None = None
        self._connected: bool = False
        self._req_counter: int = 0
        self._lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        # What the server last wrote to stderr, kept so a crash can be explained. It is
        # drained as it arrives: an unread pipe fills at ~64 KiB and the server blocks.
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        # Set once the owner is done with this client; a tool looked up just before
        # then must not start the server again behind the owner's back.
        self._retired = False
        self._discovered_tools: dict[str, MCPTool] = {}
        self._http: httpx.AsyncClient | None = None
        self._http_session_id: str | None = None
        self._http_protocol_version: str | None = None

    @property
    def config(self) -> MCPConnectionConfig:
        """Server connection configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    def _next_req_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _build_environment(self) -> dict[str, str]:
        """Construct isolated environment filtering out secret variables."""
        child_env: dict[str, str] = {}

        # 1. Allowlisted host environment variables (strictly filtering secret patterns)
        for key in self._config.env_allowlist:
            if is_secret_env_name(key):
                # Threat model D1 / Principle 6: Secrets cannot be implicitly inherited via allowlist
                continue
            if key in os.environ:
                child_env[key] = os.environ[key]

        # 2. Explicitly passed environment variables (explicit grants at call site)
        for key, value in self._config.env.items():
            child_env[key] = str(value)

        return child_env

    def _validate_network_policy(self) -> None:
        """Validate network egress policy for remote MCP servers."""
        if not self._config.effective_allow_network:
            raise SandboxViolationError(
                f"MCP server '{self._config.server_name}' uses network transport "
                f"'{self._config.transport}', but allow_network=False"
            )
        if not self._config.url:
            raise ValueError(
                f"MCP server '{self._config.server_name}' requires a 'url' for "
                f"'{self._config.transport}' transport"
            )
        if self._config.effective_egress_allowlist:
            parsed = urllib.parse.urlparse(self._config.url)
            host = parsed.hostname or self._config.url
            matched = False
            for pattern in self._config.effective_egress_allowlist:
                if (
                    host == pattern
                    or fnmatch.fnmatch(host, pattern)
                    or fnmatch.fnmatch(self._config.url, pattern)
                ):
                    matched = True
                    break
            if not matched:
                raise SandboxViolationError(
                    f"Egress to '{self._config.url}' is forbidden by egress_allowlist: "
                    f"{self._config.effective_egress_allowlist}"
                )

    def _validate_stdio_policy(self) -> Path | None:
        """Validate boundaries and command path for local STDIO subprocess."""
        if not self._config.command:
            raise ValueError(
                f"MCP server '{self._config.server_name}' requires 'command' for STDIO transport"
            )

        workspace_root = self._config.workspace_root
        isolation = self._config.isolation

        if isolation.level == IsolationLevel.WORKSPACE:
            if workspace_root is not None:
                safe_root = self._validator.resolve_safe_path(Path("."), workspace_root)
                p_cmd = Path(self._config.command)
                if ".." in p_cmd.parts:
                    self._validator.resolve_safe_path(p_cmd, safe_root)

                for arg in self._config.args:
                    if ".." in arg:
                        self._validator.resolve_safe_path(Path(arg), safe_root)

                if isolation.write_paths:
                    for wp in isolation.write_paths:
                        self._validator.resolve_safe_path(wp, safe_root)
                return safe_root
            else:
                p_cmd = Path(self._config.command)
                if ".." in p_cmd.parts:
                    raise SandboxViolationError(
                        f"Command '{self._config.command}' contains path traversal without workspace_root"
                    )
        return workspace_root

    def _validate_argument_paths(self, arguments: dict[str, Any], workspace_root: Path) -> None:
        """Verify that any path-like arguments resolve strictly within workspace boundaries."""
        for key, value in arguments.items():
            if isinstance(value, str):
                key_lower = key.lower()
                is_path_key = any(
                    k in key_lower
                    for k in (
                        "path",
                        "file",
                        "dir",
                        "cwd",
                        "target",
                        "dest",
                        "src",
                        "output",
                        "input",
                    )
                )
                if is_path_key or ".." in value or value.startswith("/"):
                    p = Path(value)
                    self._validator.resolve_safe_path(p, workspace_root)
            elif isinstance(value, dict):
                raw_dict = cast(dict[object, object], value)
                sub_dict: dict[str, Any] = {str(k): v for k, v in raw_dict.items()}
                self._validate_argument_paths(sub_dict, workspace_root)
            elif isinstance(value, (list, tuple)):
                items = cast(Sequence[object], value)
                for item in items:
                    if isinstance(item, str) and (".." in item or item.startswith("/")):
                        self._validator.resolve_safe_path(Path(item), workspace_root)
                    elif isinstance(item, dict):
                        raw_dict2 = cast(dict[object, object], item)
                        sub_dict2: dict[str, Any] = {str(k): v for k, v in raw_dict2.items()}
                        self._validate_argument_paths(sub_dict2, workspace_root)

    async def _send_jsonrpc_request(
        self, request_payload: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Send one request and return the response carrying its id.

        One request at a time (`_request_lock`): replies are matched by reading, so two
        concurrent callers on one pipe would each be able to take the other's reply.
        """
        async with self._request_lock:
            if self._config.transport == MCPTransport.STREAMABLE_HTTP:
                return await asyncio.wait_for(
                    self._http_request(request_payload), timeout=timeout_seconds
                )
            return await asyncio.wait_for(
                self._stdio_request(request_payload), timeout=timeout_seconds
            )

    async def _stdio_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(
                f"MCP server process for '{self._config.server_name}' is not running"
            )

        req_bytes = (json.dumps(request_payload) + "\n").encode("utf-8")
        self._process.stdin.write(req_bytes)
        await self._process.stdin.drain()

        want_id = request_payload.get("id")
        while True:
            line_bytes = await self._process.stdout.readline()
            if not line_bytes:
                await asyncio.sleep(0.05)  # let the drain collect a dying server's last words
                stderr_out = self._stderr_tail.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"MCP server closed stream unexpectedly. Stderr: {stderr_out}")
            message = _parse_jsonrpc_message(line_bytes.decode("utf-8", errors="replace"))
            if message is None:
                continue  # a log line some servers print to stdout; not a message
            if await self._answer_server_request(message):
                continue
            # A notification (`notifications/message`, progress) arrives on the same pipe
            # and has no id; taking it for the reply returned an empty result.
            if message.get("id") == want_id and ("result" in message or "error" in message):
                return message

    async def _answer_server_request(self, message: dict[str, Any]) -> bool:
        """Reply to a request the server sent us. Returns False if `message` is not one.

        `ping` is answered, as the protocol requires; anything else (sampling, roots) is a
        capability this client never declared, so it is refused rather than left hanging.
        """
        if "method" not in message or "id" not in message:
            return False
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if message.get("method") == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {"code": -32601, "message": "Method not supported by client"}
        await self._send_jsonrpc_notification(reply)
        return True

    async def _http_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """POST one JSON-RPC request (Streamable HTTP) and read its reply.

        The server may answer with a single JSON body or with an SSE stream that carries
        the reply among other messages; both are accepted, as the transport requires.
        """
        http = self._http
        if http is None or not self._config.url:
            raise RuntimeError(f"MCP server '{self._config.server_name}' is not connected")
        try:
            return await self._http_exchange(http, self._config.url, request_payload)
        except httpx.LocalProtocolError as exc:
            # h11 quotes the offending header value in its message, and a header here is
            # usually a credential; the type is all that may leave.
            raise RuntimeError(
                f"MCP server '{self._config.server_name}': a configured header could not be "
                "sent (it may contain a line break). Re-enter its value."
            ) from exc

    async def _http_exchange(
        self, http: httpx.AsyncClient, url: str, request_payload: dict[str, Any]
    ) -> dict[str, Any]:
        want_id = request_payload.get("id")
        async with http.stream(
            "POST", url, json=request_payload, headers=self._http_headers()
        ) as response:
            if response.is_redirect:
                # Redirects are not followed: httpx drops only `Authorization` when one
                # leaves the origin, so any other sign-in header would go to the new host.
                location = response.headers.get("location", "")
                raise RuntimeError(
                    f"The server redirected to {location!r} (HTTP {response.status_code}). "
                    "Use that address instead."
                )
            if response.status_code >= 400:
                body = (await _read_bounded(response, 4096)).decode("utf-8", errors="replace")
                raise RuntimeError(_http_error_text(response.status_code, body[:300]))
            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self._http_session_id = session_id
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                data_lines: list[str] = []
                data_size = 0
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        data_size += len(line)
                        if data_size > _MAX_HTTP_MESSAGE:
                            raise RuntimeError(_too_large_text(self._config.server_name))
                        continue
                    if line == "" and data_lines:
                        message = _parse_jsonrpc_message("\n".join(data_lines))
                        data_lines = []
                        if message is not None and message.get("id") == want_id:
                            return message
                raise RuntimeError(
                    f"MCP server '{self._config.server_name}' ended the event stream "
                    "without answering"
                )
            raw = await _read_bounded(response, _MAX_HTTP_MESSAGE)
            if len(raw) > _MAX_HTTP_MESSAGE:
                raise RuntimeError(_too_large_text(self._config.server_name))
            parsed = cast(object, json.loads(raw.decode("utf-8")))
            candidates = cast(list[object], parsed) if isinstance(parsed, list) else [parsed]
            for item in candidates:
                if isinstance(item, dict):
                    message = {str(k): v for k, v in cast(dict[object, object], item).items()}
                    if message.get("id") == want_id:
                        return message
            raise RuntimeError(
                f"MCP server '{self._config.server_name}' answered without a reply to the request"
            )

    def _http_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **dict(self._config.headers),
        }
        if self._http_session_id:
            headers["Mcp-Session-Id"] = self._http_session_id
        if self._http_protocol_version:
            headers["MCP-Protocol-Version"] = self._http_protocol_version
        return headers

    async def _send_jsonrpc_notification(self, payload: dict[str, Any]) -> None:
        if self._config.transport == MCPTransport.STREAMABLE_HTTP:
            if self._http is None or not self._config.url:
                return
            response = await self._http.post(
                self._config.url, json=payload, headers=self._http_headers()
            )
            if response.status_code >= 400:
                raise RuntimeError(_http_error_text(response.status_code, response.text[:300]))
            return
        if self._process is None or self._process.stdin is None:
            return
        notif_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        self._process.stdin.write(notif_bytes)
        await self._process.stdin.drain()

    async def _initialize(self, protocol_version: str) -> None:
        init_req: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_req_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "uclone-x",
                    # Read through the module, not
                    # `from uclone_x import __version__`: an
                    # import-time binding reports the same string
                    # but cannot be moved by a test, so nothing
                    # could show this handshake follows the
                    # declaration (#1131, following #1121).
                    "version": uclone_x.__version__,
                },
            },
        }
        resp = await self._send_jsonrpc_request(init_req, timeout_seconds=self._connect_timeout)
        if "error" in resp:
            raise RuntimeError(
                f"MCP server '{self._config.server_name}' refused to initialize: "
                f"{_jsonrpc_error_text(resp.get('error'))}"
            )
        result = resp.get("result")
        if isinstance(result, dict):
            agreed = cast(dict[str, object], result).get("protocolVersion")
            if isinstance(agreed, str) and agreed:
                self._http_protocol_version = (
                    agreed if self._config.transport == MCPTransport.STREAMABLE_HTTP else None
                )
        notif: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        await self._send_jsonrpc_notification(notif)

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        async with self._lock:
            if self._connected:
                return
            if self._retired:
                raise RuntimeError(
                    f"MCP server '{self._config.server_name}' was removed or turned off"
                )

            if self._config.transport == MCPTransport.STREAMABLE_HTTP:
                self._validate_network_policy()
                self._http = httpx.AsyncClient(
                    timeout=httpx.Timeout(self._connect_timeout, read=None),
                    follow_redirects=False,
                    transport=self._http_transport,
                )
                try:
                    await self._initialize(_HTTP_PROTOCOL_VERSION)
                except Exception:
                    await self._close_http()
                    raise
                self._connected = True
                return

            if self._config.transport in (MCPTransport.SSE, MCPTransport.WEBSOCKET):
                self._validate_network_policy()
                self._connected = True
                return

            if self._config.transport == MCPTransport.STDIO:
                safe_root = self._validate_stdio_policy()
                child_env = self._build_environment()
                cwd_str = str(safe_root) if safe_root is not None else None

                assert self._config.command is not None
                # A local server is a process the model drives, so it runs where the
                # system allows in the jail that keeps it from writing the story library,
                # as `bash_run` does (#1589). `sandbox-exec` starts the command itself,
                # so a missing command is looked for first: otherwise it would surface as
                # `sandbox-exec` failing, not as a missing server.
                jail = story_library_jail(self._config.workspace_root)
                if jail and not _command_exists(self._config.command, child_env, cwd_str):
                    raise FileNotFoundError(
                        f"MCP server executable not found: '{self._config.command}'"
                    )
                try:
                    self._process = await asyncio.create_subprocess_exec(
                        *jail,
                        self._config.command,
                        *self._config.args,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd_str,
                        env=child_env,
                        # A server's stdout line can be a whole tool listing; asyncio's
                        # 64 KiB default makes `readline` fail on a large one.
                        limit=_STDIO_LINE_LIMIT,
                        # Its own process group, so stopping it also stops what a launcher
                        # (`npx`, `uvx`) started beneath it.
                        start_new_session=os.name == "posix",
                    )
                except FileNotFoundError as err:
                    raise FileNotFoundError(
                        f"MCP server executable not found: '{self._config.command}'"
                    ) from err
                self._stderr_tail.clear()
                self._stderr_task = asyncio.create_task(self._drain_stderr(self._process))

                try:
                    await self._initialize(_STDIO_PROTOCOL_VERSION)
                except Exception:
                    await self._terminate_process()
                    raise

                self._connected = True

    async def _close_http(self) -> None:
        http, self._http = self._http, None
        if http is None:
            return
        if self._http_session_id and self._config.url:
            # Ending the session is a courtesy the transport asks for; a server that
            # does not support it answers 405, and one that is gone cannot be told.
            try:
                await http.delete(self._config.url, headers=self._http_headers(), timeout=2.0)
            except Exception:
                pass
        self._http_session_id = None
        self._http_protocol_version = None
        await http.aclose()

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        stream = process.stderr
        if stream is None:
            return
        while chunk := await stream.read(4096):
            self._stderr_tail += chunk
            del self._stderr_tail[:-_STDERR_TAIL]

    @staticmethod
    def _signal_process(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, sig)
                return
            except (ProcessLookupError, PermissionError):
                pass
        process.send_signal(sig)

    async def _terminate_process(self) -> None:
        if self._process is not None:
            try:
                if self._process.stdin and not self._process.stdin.is_closing():
                    self._process.stdin.close()
                self._signal_process(self._process, signal.SIGTERM)
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except (ProcessLookupError, TimeoutError):
                try:
                    self._signal_process(self._process, signal.SIGKILL)
                    await self._process.wait()
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self._process = None
                task, self._stderr_task = self._stderr_task, None
                if task is not None:
                    task.cancel()

    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        async with self._lock:
            await self._terminate_process()
            await self._close_http()
            self._connected = False
            self._discovered_tools.clear()

    async def retire(self) -> None:
        """Disconnect for good: later calls through this client's tools are refused."""
        self._retired = True
        await self.disconnect()

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.disconnect()

    def _supports_requests(self) -> bool:
        return self._config.transport in (MCPTransport.STDIO, MCPTransport.STREAMABLE_HTTP)

    async def list_tools(self) -> list[ToolProtocol]:
        """Discover tools exposed by the MCP server, following pagination."""
        if not self._connected:
            await self.connect()

        if not self._supports_requests():
            raise NotImplementedError(
                f"MCP transport '{self._config.transport.value}' is not implemented"
            )

        tools: list[ToolProtocol] = []
        cursor: str | None = None
        for _page in range(_MAX_TOOL_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            req: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": self._next_req_id(),
                "method": "tools/list",
                "params": params,
            }
            resp = await self._send_jsonrpc_request(req, timeout_seconds=self._connect_timeout)
            if "error" in resp:
                raise RuntimeError(
                    f"MCP server '{self._config.server_name}' refused to list its tools: "
                    f"{_jsonrpc_error_text(resp.get('error'))}"
                )
            result_obj = resp.get("result")
            result_data: dict[str, Any] = (
                {str(k): v for k, v in cast(dict[object, object], result_obj).items()}
                if isinstance(result_obj, dict)
                else {}
            )
            tools_list_obj = result_data.get("tools")
            tools_list: list[Any] = (
                list(cast(Sequence[object], tools_list_obj))
                if isinstance(tools_list_obj, (list, tuple))
                else []
            )
            for t_item in tools_list:
                t: dict[str, Any] = (
                    {str(k): v for k, v in cast(dict[object, object], t_item).items()}
                    if isinstance(t_item, dict)
                    else {}
                )
                remote_name = str(t.get("name", ""))
                if not remote_name:
                    continue
                tool_desc = str(t.get("description", "") or "")
                schema_raw = t.get("inputSchema")
                schema: dict[str, Any] = (
                    {str(k): v for k, v in cast(dict[object, object], schema_raw).items()}
                    if isinstance(schema_raw, dict)
                    else {"type": "object", "properties": {}}
                )
                exposed = exposed_tool_name(self._tool_name_prefix, remote_name)
                mcp_tool = MCPTool(
                    name=exposed,
                    description=tool_desc,
                    parameters_schema=schema,
                    client=self,
                    remote_name=remote_name,
                )
                self._discovered_tools[exposed] = mcp_tool
                tools.append(mcp_tool)
            next_cursor = result_data.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Invoke a tool on the MCP server. `name` is the server's own name for it."""
        start_time = time.monotonic()

        if self._retired:
            return ToolResult(
                success=False,
                error=(
                    f"The tool server '{self._config.server_name}' was removed or turned off "
                    "in Settings, so its tools cannot be used."
                ),
                provenance=Provenance.primary(
                    provider=f"mcp.{self._config.server_name}", model=name
                ),
            )

        if not self._connected:
            await self.connect()

        effective_level = effective_isolation_level(
            context.isolation.level, self._config.isolation.level
        )

        safe_workspace = self._validator.resolve_safe_path(Path("."), context.require_workspace())

        if effective_level is not IsolationLevel.NONE:
            self._validate_argument_paths(arguments, safe_workspace)

        prov = Provenance.primary(
            provider=f"mcp.{self._config.server_name}",
            model=name,
        )

        if not self._supports_requests():
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"MCP transport '{self._config.transport.value}' is not implemented",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=effective_level,
                provenance=prov,
            )

        req: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_req_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        try:
            resp = await self._send_jsonrpc_request(req, timeout_seconds=context.timeout_seconds)
            duration_ms = (time.monotonic() - start_time) * 1000.0

            if "error" in resp:
                return ToolResult(
                    success=False,
                    error=_jsonrpc_error_text(resp.get("error")),
                    execution_time_ms=round(duration_ms, 3),
                    isolation_level=effective_level,
                    provenance=prov,
                )

            result_data = resp.get("result")
            is_error: bool = False
            output_data: JsonValue = None
            error_text: str | None = None

            if isinstance(result_data, dict):
                res_map = cast(dict[str, object], result_data)
                is_error = bool(res_map.get("isError", False))
                content_obj: object = res_map["content"] if "content" in res_map else res_map
                if is_error:
                    error_text = str(content_obj)
                else:
                    output_data = cast(JsonValue, content_obj)
            else:
                output_data = cast(JsonValue, result_data)

            return ToolResult(
                success=not is_error,
                output=output_data,
                error=error_text,
                execution_time_ms=round(duration_ms, 3),
                isolation_level=effective_level,
                provenance=prov,
            )
        except TimeoutError:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Tool '{name}' timed out after {context.timeout_seconds}s",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=effective_level,
                provenance=prov,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {exc}",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=effective_level,
                provenance=prov,
            )


def _parse_jsonrpc_message(text: str) -> dict[str, Any] | None:
    """One JSON-RPC message object, or None for anything that is not one."""
    try:
        parsed = cast(object, json.loads(text))
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(k): v for k, v in cast(dict[object, object], parsed).items()}


def _jsonrpc_error_text(err_info: object) -> str:
    if isinstance(err_info, dict):
        err_map = cast(dict[str, object], err_info)
        msg_val = err_map.get("message")
        return str(msg_val) if msg_val is not None else str(err_map)
    return str(err_info)


def _http_error_text(status: int, body: str) -> str:
    """Name the cause of a refused HTTP request, and for 401/403 what would fix it."""
    if status in (401, 403):
        return (
            f"The server refused access (HTTP {status}). It likely needs a sign-in header "
            "such as 'Authorization: Bearer <token>'."
        )
    if status == 404:
        return f"Nothing answers at this address (HTTP 404). Check the URL. {body}".strip()
    return f"The server answered HTTP {status}. {body}".strip()


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    """Read at most `limit + 1` bytes, so the caller can tell an oversize body apart."""
    out = bytearray()
    async for chunk in response.aiter_bytes():
        out += chunk
        if len(out) > limit:
            break
    return bytes(out[: limit + 1])


def _too_large_text(server_name: str) -> str:
    return (
        f"MCP server '{server_name}' sent a reply larger than "
        f"{_MAX_HTTP_MESSAGE // (1024 * 1024)} MiB; it was not read."
    )


_TOOL_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def exposed_tool_name(prefix: str | None, remote_name: str) -> str:
    """The name a model is shown for an MCP tool: `<prefix>__<remote>`, provider-safe.

    Model providers accept `^[A-Za-z0-9_-]{1,64}$`; a server's own names are not bound
    by that, and two servers may both offer `search`. Without a prefix the server's name
    is used as it is, which is what configuration-file servers have always done.
    """
    if prefix is None:
        return remote_name
    return _TOOL_NAME_UNSAFE.sub("_", f"{prefix}__{remote_name}")[:64]


_STDIO_PROTOCOL_VERSION = "2024-11-05"
_HTTP_PROTOCOL_VERSION = "2025-06-18"
_STDIO_LINE_LIMIT = 16 * 1024 * 1024
# One HTTP reply (JSON body or SSE message). A tool listing is kilobytes; this bounds what a
# misbehaving server can make the app hold before the request's timeout would.
_MAX_HTTP_MESSAGE = 16 * 1024 * 1024
_STDERR_TAIL = 4096
_MAX_TOOL_PAGES = 50


# Static conformance check
_mcp_conformance: MCPClientProtocol = MCPClient(
    config=MCPConnectionConfig(
        server_name="conformance", command="python", workspace_root=Path("/tmp")
    )
)
