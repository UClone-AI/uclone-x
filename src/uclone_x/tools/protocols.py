"""Protocols for tools, tool registry, and MCP client/adapters.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from uclone_x.tools.models import MCPConnectionConfig, ToolContext, ToolResult


class ToolProtocol(Protocol):
    """Protocol representing an executable tool."""

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    def description(self) -> str:
        """Detailed functional description for LLM reasoning."""
        ...

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining tool input arguments."""
        ...

    async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with arguments and execution context."""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for managing and resolving available tools."""

    def register(self, tool: ToolProtocol) -> None:
        """Register a new tool instance."""
        ...

    def get(self, name: str) -> ToolProtocol | None:
        """Look up a tool by name."""
        ...

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns False when no tool had that name."""
        ...

    def list_tools(
        self,
        filter_names: tuple[str, ...] | None = None,
    ) -> list[ToolProtocol]:
        """List registered tools matching optional whitelist."""
        ...


class MCPClientProtocol(Protocol):
    """Protocol for communicating with an external Model Context Protocol server."""

    @property
    def config(self) -> MCPConnectionConfig:
        """Server connection configuration."""
        ...

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        ...

    async def list_tools(self) -> list[ToolProtocol]:
        """Discover tools exposed by the MCP server."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Invoke a tool on the remote MCP server."""
        ...
