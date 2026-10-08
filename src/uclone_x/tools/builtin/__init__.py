"""Built-in tool suite for UClone-X runtime."""

# Every name is re-exported lazily (PEP 562). `tools/registry.py` imports
# `tools.builtin.shell`, which runs this file first; eager re-exports here loaded every
# built-in tool with it, and `character` brings the story package (#2012).

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uclone_x.tools.builtin.character import (
        CharacterSheetParams,
        CharacterSheetTool,
    )
    from uclone_x.tools.builtin.comfy_client import (
        ComfyClient,
        ComfyTimeoutError,
        format_exec_error,
        format_node_errors,
    )
    from uclone_x.tools.builtin.comfy_image_tool import (
        ComfyImageGenParams,
        ComfyImageGenTool,
        build_txt2img_workflow,
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
    from uclone_x.tools.builtin.image import (
        GenerateImageParams,
        GenerateImageTool,
        ImagePipelineDispatcher,
    )
    from uclone_x.tools.builtin.mcp_loader import (
        MCPConfigFileLoader,
        expand_env_vars,
    )
    from uclone_x.tools.builtin.media_registry import (
        ModelProfile,
        ModelRegistry,
        PromptFamily,
    )
    from uclone_x.tools.builtin.shell import BashRunTool
    from uclone_x.tools.builtin.skill_loader import (
        LoadSkillParams,
        LoadSkillTool,
    )
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
    from uclone_x.tools.registry import (
        create_default_registry,
        create_default_tool_registry,
    )

#: Each lazily re-exported name -> the module that defines it, loaded on first access.
_LAZY_NAMES: dict[str, str] = {
    "CharacterSheetParams": "uclone_x.tools.builtin.character",
    "CharacterSheetTool": "uclone_x.tools.builtin.character",
    "ComfyClient": "uclone_x.tools.builtin.comfy_client",
    "ComfyTimeoutError": "uclone_x.tools.builtin.comfy_client",
    "format_exec_error": "uclone_x.tools.builtin.comfy_client",
    "format_node_errors": "uclone_x.tools.builtin.comfy_client",
    "ComfyImageGenParams": "uclone_x.tools.builtin.comfy_image_tool",
    "ComfyImageGenTool": "uclone_x.tools.builtin.comfy_image_tool",
    "build_txt2img_workflow": "uclone_x.tools.builtin.comfy_image_tool",
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
    "GenerateImageParams": "uclone_x.tools.builtin.image",
    "GenerateImageTool": "uclone_x.tools.builtin.image",
    "ImagePipelineDispatcher": "uclone_x.tools.builtin.image",
    "MCPConfigFileLoader": "uclone_x.tools.builtin.mcp_loader",
    "expand_env_vars": "uclone_x.tools.builtin.mcp_loader",
    "ModelProfile": "uclone_x.tools.builtin.media_registry",
    "ModelRegistry": "uclone_x.tools.builtin.media_registry",
    "PromptFamily": "uclone_x.tools.builtin.media_registry",
    "BashRunTool": "uclone_x.tools.builtin.shell",
    "LoadSkillParams": "uclone_x.tools.builtin.skill_loader",
    "LoadSkillTool": "uclone_x.tools.builtin.skill_loader",
    "DuckDuckGoSearchProvider": "uclone_x.tools.builtin.web",
    "SearchProviderProtocol": "uclone_x.tools.builtin.web",
    "WebFetchParams": "uclone_x.tools.builtin.web",
    "WebFetchTool": "uclone_x.tools.builtin.web",
    "WebSearchParams": "uclone_x.tools.builtin.web",
    "WebSearchTool": "uclone_x.tools.builtin.web",
    "html_to_markdown": "uclone_x.tools.builtin.web",
    "is_private_ip": "uclone_x.tools.builtin.web",
    "validate_url_ssrf": "uclone_x.tools.builtin.web",
    "create_default_registry": "uclone_x.tools.registry",
    "create_default_tool_registry": "uclone_x.tools.registry",
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
    "BashRunTool",
    "CharacterSheetParams",
    "CharacterSheetTool",
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
    "GenerateImageParams",
    "GenerateImageTool",
    "ImagePipelineDispatcher",
    "LoadSkillParams",
    "LoadSkillTool",
    "MCPConfigFileLoader",
    "ModelProfile",
    "ModelRegistry",
    "PromptFamily",
    "SearchProviderProtocol",
    "WebFetchParams",
    "WebFetchTool",
    "WebSearchParams",
    "WebSearchTool",
    "build_txt2img_workflow",
    "create_default_registry",
    "create_default_tool_registry",
    "expand_env_vars",
    "format_exec_error",
    "format_node_errors",
    "html_to_markdown",
    "is_private_ip",
    "validate_url_ssrf",
]
