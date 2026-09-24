"""Agent Client Protocol (ACP) subsystem.

Today this package holds only the conformance registry — the single source for what we claim
to answer, checked against `docs/acp-protocol-spec.md`. The stdio JSON-RPC shell itself is
#649 and lands here beside it.
"""

from uclone_x.acp.conformance import (
    ACP_MCP_LOADER_WARNING,
    ACP_SDK_VERSION,
    ACP_TRANSPORT,
    AcpConformanceReport,
    AcpMcpDescriptor,
    AcpMethod,
    AcpMethodStatus,
    AcpShellPresence,
    AcpSide,
    AcpSideCounts,
    acp_mcp_descriptor_registry,
    acp_method_registry,
    conformance_summary,
    implemented,
    shell_presence,
)

__all__ = [
    "ACP_MCP_LOADER_WARNING",
    "ACP_SDK_VERSION",
    "ACP_TRANSPORT",
    "AcpConformanceReport",
    "AcpMcpDescriptor",
    "AcpMethod",
    "AcpMethodStatus",
    "AcpShellPresence",
    "AcpSide",
    "AcpSideCounts",
    "acp_mcp_descriptor_registry",
    "acp_method_registry",
    "conformance_summary",
    "implemented",
    "shell_presence",
]
