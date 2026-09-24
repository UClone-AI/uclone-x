"""Tool registry and local tool implementations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from uclone_x.tools.builtin.shell import BashRunTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "LocalTool",
    "ToolRegistry",
    "create_default_registry",
    "create_default_tool_registry",
]


class LocalTool:
    """A locally executable tool adhering to ToolProtocol."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any] | None = None,
        handler: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]] | None = None,
        *,
        writes_files: bool = True,
    ) -> None:
        """Wrap `handler` as a tool.

        `writes_files` says whether the handler can create, modify or delete a file on the
        host; see `uclone_x.tools.base.tool_writes_files`. It defaults to `True` because a
        handler is opaque: one that is refused under `enable_write_tools: false` until
        someone declares it read-only is safe, one that is trusted until it writes is not.
        """
        self.writes_files = writes_files
        self._name = name
        self._description = description
        self._parameters_schema: dict[str, Any] = (
            parameters_schema
            if parameters_schema is not None
            else {"type": "object", "properties": {}}
        )
        self._handler = handler

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
        """Execute the tool with arguments and execution context."""
        if self._handler is not None:
            return await self._handler(params, context)

        raise RuntimeError(f"Tool '{self._name}' has no execution handler configured")


class ToolRegistry:
    """Registry for managing and resolving available tools."""

    def __init__(self, tools: Sequence[ToolProtocol] | None = None) -> None:
        self._tools: dict[str, ToolProtocol] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(self, tool: ToolProtocol) -> None:
        """Register a new tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolProtocol | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name, as when its MCP server is disconnected or deleted."""
        return self._tools.pop(name, None) is not None

    def list_tools(
        self,
        filter_names: tuple[str, ...] | None = None,
    ) -> list[ToolProtocol]:
        """List registered tools matching optional whitelist."""
        if filter_names is None:
            return list(self._tools.values())
        name_set = set(filter_names)
        return [tool for tool in self._tools.values() if tool.name in name_set]

    @classmethod
    def with_builtins(
        cls,
        workspace_root: Path | None = None,
        enable_mcp: bool = True,
        mcp_config_path: Path | None = None,
    ) -> ToolRegistry:
        """Create a registry pre-populated with standard builtin tools."""
        return create_default_registry(
            workspace_root=workspace_root,
            enable_mcp=enable_mcp,
            mcp_config_path=mcp_config_path,
        )

    @classmethod
    def default(
        cls,
        workspace_root: Path | None = None,
        enable_mcp: bool = True,
        mcp_config_path: Path | None = None,
    ) -> ToolRegistry:
        """Create a default tool registry pre-populated with built-in tools."""
        return create_default_registry(
            workspace_root=workspace_root,
            enable_mcp=enable_mcp,
            mcp_config_path=mcp_config_path,
        )


def create_default_registry(
    workspace_root: Path | None = None,
    enable_mcp: bool = True,
    mcp_config_path: Path | None = None,
) -> ToolRegistry:
    """Create a ToolRegistry populated with the default builtin tool suite and configured MCP tools."""
    from uclone_x.tools.builtin.character import CharacterSheetTool
    from uclone_x.tools.builtin.filesystem import (
        DirectoryListTool,
        FileEditTool,
        FileReadTool,
        FileSearchTool,
        FileWriteTool,
    )
    from uclone_x.tools.builtin.image import GenerateImageTool
    from uclone_x.tools.builtin.install import InstallPackageTool
    from uclone_x.tools.builtin.mcp_loader import MCPConfigFileLoader
    from uclone_x.tools.builtin.plan import PlanUpdateTool
    from uclone_x.tools.builtin.subagent import SubagentDelegationTool
    from uclone_x.tools.builtin.tool_results import ToolResultReadTool
    from uclone_x.tools.builtin.web import (
        WebFetchTool,
        WebSearchTool,
    )

    registry = ToolRegistry(
        tools=[
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            FileSearchTool(),
            DirectoryListTool(),
            BashRunTool(name="bash_run"),
            # The same shell under a second name, kept registered so a stored call or a
            # persona file naming it still resolves. Not advertised beside `bash_run`:
            # two identical schemas were ~1 KB per request and a coin flip for the model
            # (#1424).
            BashRunTool(name="run_command", alias_of="bash_run"),
            WebFetchTool(),
            WebSearchTool(),
            GenerateImageTool(),
            CharacterSheetTool(),
            InstallPackageTool(),
            PlanUpdateTool(),
            SubagentDelegationTool(),
            ToolResultReadTool(),
        ]
    )

    if enable_mcp:
        loader = MCPConfigFileLoader(
            workspace_root=workspace_root,
            config_path=mcp_config_path,
        )
        try:
            loader.load_and_register_tools(registry)
        except Exception as exc:
            logger.warning("Failed to auto-wire MCP tools into default registry: %s", exc)

    return registry


create_default_tool_registry = create_default_registry


# Static protocol conformance check
_registry_conformance: ToolRegistryProtocol = ToolRegistry()
_tool_conformance: ToolProtocol = LocalTool(name="test", description="test")
_bash_tool_conformance: ToolProtocol = BashRunTool()
