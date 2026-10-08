"""Tools & MCP subsystem: Local tool runtime, tool registry, BaseTool, and MCP client."""

# The built-in tool adapters are re-exported lazily (PEP 562), as `uclone_x.llm` does its
# connectors (#1775). Importing any module in this package runs this file first, so the
# eager re-exports made `import uclone_x.tools.models` -- a contract the agent kernel
# reads -- load every built-in tool, and through `tools.builtin` the story package
# (#2012). `from uclone_x.tools import WebFetchTool` still works, on first use.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from uclone_x.tools.base import BaseTool
from uclone_x.tools.client import (
    MCPClient,
    MCPTool,
)
from uclone_x.tools.models import (
    SECRET_ENV_PATTERNS,
    IsolationLevel,
    IsolationPolicy,
    MCPConnectionConfig,
    MCPTransport,
    NoIsolation,
    ToolContext,
    ToolResult,
    ToolResultStatus,
    WorkspaceIsolation,
    effective_isolation_level,
    is_secret_env_name,
)
from uclone_x.tools.protocols import (
    MCPClientProtocol,
    ToolProtocol,
    ToolRegistryProtocol,
)
from uclone_x.tools.registry import (
    LocalTool,
    ToolRegistry,
    create_default_registry,
    create_default_tool_registry,
)
from uclone_x.tools.scoped_registry import ScopedToolRegistry

if TYPE_CHECKING:
    from uclone_x.tools.builtin.comfy_client import (
        ComfyClient,
        ComfyTimeoutError,
    )
    from uclone_x.tools.builtin.comfy_image_tool import (
        ComfyImageGenParams,
        ComfyImageGenTool,
    )
    from uclone_x.tools.builtin.filesystem import (
        DirectoryListParams,
        DirectoryListTool,
        FileEditParams,
        FileEditTool,
        FileReadParams,
        FileReadTool,
        FileSearchParams,
        FileSearchTool,
        FileWriteParams,
        FileWriteTool,
    )
    from uclone_x.tools.builtin.mcp_loader import (
        MCPConfigFileLoader,
        expand_env_vars,
    )
    from uclone_x.tools.builtin.shell import BashRunTool
    from uclone_x.tools.builtin.web import (
        DuckDuckGoSearchProvider,
        SearchProviderProtocol,
        WebFetchParams,
        WebFetchTool,
        WebSearchParams,
        WebSearchTool,
        html_to_markdown,
        is_private_ip,
        validate_url_ssrf,
    )

#: Each lazily re-exported name -> the module that defines it, loaded on first access.
_LAZY_NAMES: dict[str, str] = {
    "ComfyClient": "uclone_x.tools.builtin.comfy_client",
    "ComfyTimeoutError": "uclone_x.tools.builtin.comfy_client",
    "ComfyImageGenParams": "uclone_x.tools.builtin.comfy_image_tool",
    "ComfyImageGenTool": "uclone_x.tools.builtin.comfy_image_tool",
    "DirectoryListParams": "uclone_x.tools.builtin.filesystem",
    "DirectoryListTool": "uclone_x.tools.builtin.filesystem",
    "FileEditParams": "uclone_x.tools.builtin.filesystem",
    "FileEditTool": "uclone_x.tools.builtin.filesystem",
    "FileReadParams": "uclone_x.tools.builtin.filesystem",
    "FileReadTool": "uclone_x.tools.builtin.filesystem",
    "FileSearchParams": "uclone_x.tools.builtin.filesystem",
    "FileSearchTool": "uclone_x.tools.builtin.filesystem",
    "FileWriteParams": "uclone_x.tools.builtin.filesystem",
    "FileWriteTool": "uclone_x.tools.builtin.filesystem",
    "MCPConfigFileLoader": "uclone_x.tools.builtin.mcp_loader",
    "expand_env_vars": "uclone_x.tools.builtin.mcp_loader",
    "BashRunTool": "uclone_x.tools.builtin.shell",
    "DuckDuckGoSearchProvider": "uclone_x.tools.builtin.web",
    "SearchProviderProtocol": "uclone_x.tools.builtin.web",
    "WebFetchParams": "uclone_x.tools.builtin.web",
    "WebFetchTool": "uclone_x.tools.builtin.web",
    "WebSearchParams": "uclone_x.tools.builtin.web",
    "WebSearchTool": "uclone_x.tools.builtin.web",
    "html_to_markdown": "uclone_x.tools.builtin.web",
    "is_private_ip": "uclone_x.tools.builtin.web",
    "validate_url_ssrf": "uclone_x.tools.builtin.web",
}


def __getattr__(name: str) -> Any:
    """Load an adapter name on first access, so importing this package loads no adapter."""
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "BaseTool",
    "BashRunTool",
    "ComfyClient",
    "ComfyImageGenParams",
    "ComfyImageGenTool",
    "ComfyTimeoutError",
    "DirectoryListParams",
    "DirectoryListTool",
    "DuckDuckGoSearchProvider",
    "FileEditParams",
    "FileEditTool",
    "FileReadParams",
    "FileReadTool",
    "FileSearchParams",
    "FileSearchTool",
    "FileWriteParams",
    "FileWriteTool",
    "IsolationLevel",
    "IsolationPolicy",
    "LocalTool",
    "MCPClient",
    "MCPClientProtocol",
    "MCPConfigFileLoader",
    "MCPConnectionConfig",
    "MCPTool",
    "MCPTransport",
    "NoIsolation",
    "SECRET_ENV_PATTERNS",
    "ScopedToolRegistry",
    "SearchProviderProtocol",
    "ToolContext",
    "ToolProtocol",
    "ToolRegistry",
    "ToolRegistryProtocol",
    "ToolResult",
    "ToolResultStatus",
    "WebFetchParams",
    "WebFetchTool",
    "WebSearchParams",
    "WebSearchTool",
    "WorkspaceIsolation",
    "create_default_registry",
    "create_default_tool_registry",
    "effective_isolation_level",
    "expand_env_vars",
    "html_to_markdown",
    "is_private_ip",
    "is_secret_env_name",
    "validate_url_ssrf",
]
