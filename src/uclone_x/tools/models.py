"""Data models for tools and MCP (Model Context Protocol) integration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from uclone_x.core.immutable import ImmutableStrMapping
from uclone_x.core.provenance import Provenance
from uclone_x.sandbox.models import (
    SECRET_ENV_PATTERNS,
    ContainerIsolation,
    IsolationLevel,
    IsolationPolicy,
    NoIsolation,
    WasmIsolation,
    WorkspaceIsolation,
    effective_isolation_level,
    is_secret_env_name,
)

__all__ = [
    "IsolationLevel",
    "IsolationPolicy",
    "MCPConnectionConfig",
    "MCPTransport",
    "NoIsolation",
    "SECRET_ENV_PATTERNS",
    "ToolContext",
    "ToolResult",
    "ToolResultStatus",
    "WorkspaceIsolation",
    "effective_isolation_level",
    "is_secret_env_name",
]


class ToolResultStatus(StrEnum):
    """Status indicator for tool execution outcome."""

    SUCCESS = "success"
    ERROR = "error"


class MCPTransport(StrEnum):
    """Transport used to reach an MCP server."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "http"
    WEBSOCKET = "websocket"


class ToolContext(BaseModel):
    """Execution context passed to a tool invocation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, arbitrary_types_allowed=True
    )

    agent_id: str
    session_id: str
    trace_id: str | None = None
    workspace_root: Path | None = Field(
        default=None,
        description="Optional workspace root. Tools requiring it must check for its presence.",
    )
    read_roots: tuple[Path, ...] = Field(
        default=(),
        description="Folders outside the workspace that read-only file tools may also read, "
        "by absolute path. No tool may write under them.",
    )
    skill_dirs: tuple[Path, ...] = Field(
        default=(),
        description="The package folders of the skills active for the calling agent, read "
        "from its skill registry for every step, so a skill approved mid-conversation "
        "counts from the next step. A tool that reads data a skill may carry -- the story "
        "structure templates and muse tables (#1572) -- looks in them. Empty for an agent "
        "with no skill registry.",
    )

    room_id: str | None = Field(
        default=None,
        description="The conversation this call runs in: the room's id, the same for every "
        "seat in it. `session_id` is each seat's own, so it cannot say that two seats are "
        "in one conversation. It is the identity a story's writing lease is checked "
        "against (#1555). An agent another persona asked with `a2a_call` has no room of "
        "its own: its calls carry the caller's room id here, sent with the call and set on "
        "the called agent's turn (#1558), so its story writes are checked as that "
        "conversation's. `None` outside a conversation, and for an agent asked from outside "
        "one.",
    )
    story_id: str | None = Field(
        default=None,
        description="The story the conversation has open, filled from the room for every "
        "seat (#1555) and, for an agent asked with `a2a_call`, from its caller's call "
        "(#1558). Story tools compute their paths from it and take none from the "
        "model; `None` means no story is open, and they refuse with that reason.",
    )

    approved_by_person: bool = Field(
        default=False,
        description="True only when a person approved this very call through the runtime's "
        "approval request (#1557). Set by the agent after the answer arrives, never from "
        "the call's arguments, so a tool whose action needs approval "
        "(`tool_call_needs_approval`) can refuse a call nobody approved on any path.",
    )

    def require_workspace(self) -> Path:
        """Assert and return the workspace root, failing if none was provided."""
        if self.workspace_root is None:
            raise RuntimeError("Tool requires a workspace, but the host provided none.")
        return self.workspace_root

    isolation: IsolationPolicy = Field(
        default_factory=WorkspaceIsolation,
        description="Defaults to `workspace` per P3 as amended for issue "
        "2026-09-02-001; `none` is reachable only by stating `NoIsolation()`.",
    )
    timeout_seconds: float = 30.0
    turn_index: int = Field(
        default=0,
        description="The session's turn index when this tool was executed.",
    )
    agent_delegate: Any | None = Field(
        default=None,
        exclude=True,
        description="Backreference to the executing agent for tools that require agent delegation.",
    )


class ToolResult(BaseModel):
    """Result of a tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    success: bool
    output: JsonValue = None
    error: str | None = None

    execution_time_ms: float = 0.0
    artifacts: tuple[str, ...] = Field(default_factory=tuple)
    isolation_level: IsolationLevel | None = Field(
        default=None,
        description="The isolation level actually applied during execution, unifying "
        "isolation provenance with ExecutionResult.",
    )
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )

    @property
    def status(self) -> ToolResultStatus:
        """Single source of truth for tool result status."""
        return ToolResultStatus.SUCCESS if self.success else ToolResultStatus.ERROR

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        if self.success and self.error is not None and self.error.strip() != "":
            raise ValueError("ToolResult cannot have success=True with a non-empty error")
        return self


class MCPConnectionConfig(BaseModel):
    """Configuration for connecting to an external MCP server."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    server_name: str
    transport: MCPTransport = MCPTransport.STDIO
    command: str | None = None
    args: tuple[str, ...] = Field(default_factory=tuple)
    url: str | None = None
    headers: ImmutableStrMapping = Field(
        default_factory=dict,
        description="HTTP headers sent with every request to a remote (`http`) server, such "
        "as `Authorization`. Values are credentials: they are never echoed by the dashboard.",
    )
    env: ImmutableStrMapping = Field(default_factory=dict)
    env_allowlist: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Host environment names to copy in for local MCP processes. Deny-by-default: "
        "anything not named here is absent from child. Credential-shaped names matching "
        "SECRET_ENV_PATTERNS are never implicitly inherited.",
    )
    workspace_root: Path | None = Field(
        default=None,
        description="Optional workspace boundary directory for local MCP server executions.",
    )
    isolation: IsolationPolicy = Field(
        default_factory=WorkspaceIsolation,
        description="A stdio MCP server is an arbitrary local process with an arbitrary "
        "environment; before this field it sat entirely outside the sandbox model "
        "(issue 2026-09-02-042). Defaults to `workspace` like every other execution "
        "path, so an MCP provider is not privileged by omission.",
    )
    allow_network: bool = Field(
        default=False,
        description="Whether network egress is allowed for the MCP server process.",
    )
    egress_allowlist: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Allowed host/domain egress targets when allow_network is constrained.",
    )

    @property
    def effective_allow_network(self) -> bool:
        """Single source of truth for network policy."""
        if isinstance(self.isolation, (ContainerIsolation, WasmIsolation)):
            return self.isolation.allow_network
        return self.allow_network

    @property
    def effective_egress_allowlist(self) -> tuple[str, ...]:
        """Single source of truth for egress allowlist."""
        if isinstance(self.isolation, ContainerIsolation):
            return self.isolation.egress_allowlist
        return self.egress_allowlist

    @model_validator(mode="before")
    @classmethod
    def _resolve_defaults(cls, data: Any) -> Any:
        from typing import cast

        if isinstance(data, dict):
            d = cast(dict[str, Any], data)
            transport_val = d.get("transport", MCPTransport.STDIO)
            is_remote = False
            if isinstance(transport_val, MCPTransport):
                is_remote = transport_val in (MCPTransport.SSE, MCPTransport.WEBSOCKET)
            elif isinstance(transport_val, str):
                is_remote = transport_val.lower() in ("sse", "websocket")

            if not is_remote:
                ws_root = d.get("workspace_root")
                if ws_root is None:
                    iso = d.get("isolation")
                    needs_cwd = False
                    if iso is None:
                        needs_cwd = True
                    elif getattr(iso, "level", None) == IsolationLevel.WORKSPACE:
                        needs_cwd = True
                    elif isinstance(iso, str) and iso.lower() == "workspace":
                        needs_cwd = True
                    elif (
                        isinstance(iso, dict)
                        and cast(dict[str, Any], iso).get("level") == "workspace"
                    ):
                        needs_cwd = True

                    if needs_cwd:
                        data = dict(d)
                        data["workspace_root"] = Path.cwd()
        return cast(Any, data)

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        for name in self.env_allowlist:
            if is_secret_env_name(name):
                raise ValueError(
                    f"Credential-shaped environment variable '{name}' in env_allowlist is denied; "
                    "pass explicit credentials via 'env' instead"
                )

        if self.transport in (
            MCPTransport.SSE,
            MCPTransport.STREAMABLE_HTTP,
            MCPTransport.WEBSOCKET,
        ):
            if self.isolation.level != IsolationLevel.NONE:
                raise ValueError(
                    f"Remote transport '{self.transport.value}' spawns no local process and cannot "
                    f"accept filesystem-shaped isolation policy '{self.isolation.level.value}'; "
                    "use NoIsolation() explicitly for remote MCP transports"
                )
            if self.workspace_root is not None:
                raise ValueError(
                    f"Remote transport '{self.transport.value}' spawns no local process and cannot "
                    "accept 'workspace_root'; filesystem/process isolation policies apply only to "
                    "STDIO transport which spawns local processes"
                )

        if self.headers and self.transport is not MCPTransport.STREAMABLE_HTTP:
            raise ValueError(
                f"'headers' apply only to the 'http' transport, not '{self.transport.value}'"
            )

        if isinstance(self.isolation, (ContainerIsolation, WasmIsolation)):
            if self.allow_network != self.isolation.allow_network:
                raise ValueError(
                    f"MCPConnectionConfig.allow_network ({self.allow_network}) contradicts "
                    f"isolation.allow_network ({self.isolation.allow_network})"
                )

        return self
