"""Tools & MCP subsystem: Local tool runtime, tool registry, BaseTool, and MCP client."""

from uclone_x.tools.base import BaseTool
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
from uclone_x.tools.client import MCPClient, MCPTool
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
