"""The single source for what UClone-X answers of the Agent Client Protocol.

`docs/acp-protocol-spec.md` §3.3 states the requirement this module exists to meet:

> A hardcoded capability block is the mechanism by which a conformance claim becomes false
> without anyone editing a document — so the capability response must be derived from the
> same source as this table, not written in parallel with it.

So the table in §2 and §4 of that document and this registry are checked against each other
by `tests/unit/test_acp_conformance.py`, and `initialize` will derive its capability response
from `implemented()` rather than from a literal. A row cannot become *Implemented* here
without the document saying so in the same change, and vice versa.

The ACP shell itself is #649 and lands in this package alongside this module. Nothing here
starts a server, opens a transport or imports the SDK — this is the description, and the
description is deliberately readable with no ACP dependency installed.
"""

from __future__ import annotations

import importlib.util
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ACP_SDK_VERSION",
    "ACP_TRANSPORT",
    "ACP_MCP_LOADER_WARNING",
    "AcpMcpDescriptor",
    "AcpMethod",
    "AcpConformanceReport",
    "AcpMethodStatus",
    "AcpSide",
    "AcpSideCounts",
    "AcpShellPresence",
    "acp_mcp_descriptor_registry",
    "acp_method_registry",
    "conformance_summary",
    "implemented",
    "shell_presence",
]

# The version §1 of the specification names. Once ACP is a dependency, `pyproject.toml` pins
# this same string and a fitness test compares the two; until then the assertion that can be
# made is document-to-constant, and it is (see the test module). Naming a version with
# nothing checking it is how a conformance claim goes stale invisibly.
ACP_SDK_VERSION = "0.12.1"

# §6: local JSON-RPC over stdio is supported and everything else is deferred. The SDK's own
# HTTP and WebSocket transports are labelled experimental by their authors and sit behind an
# optional extra.
ACP_TRANSPORT = "stdio"


class AcpSide(StrEnum):
    """Which end of the connection calls the method."""

    AGENT = "agent"
    """A client calls this on us. The surface an ACP server shell must answer (§2)."""

    CLIENT = "client"
    """We call this on the client. The extension surface, where adoption costs fall (§4)."""


class AcpMethodStatus(StrEnum):
    """What is true of a method today.

    The four values are the specification's own vocabulary (§0), and the distinction between
    the last two is the point of §3: a method that is merely unbuilt is scheduled work, while
    a method that cannot be honoured is a design gap that must be visible before a client
    depends on it.
    """

    NOT_IMPLEMENTED = "not_implemented"
    """Specified, and scheduled. No code answers it yet."""

    IMPLEMENTED = "implemented"
    """Code exists and a test exercises it. Never set without both."""

    NOT_IMPLEMENTABLE = "not_implementable"
    """Cannot be honoured against today's architecture. A reason is required."""

    OUT_OF_SCOPE = "out_of_scope"
    """Deliberately not offered. A reason is required."""


class AcpMethod(BaseModel):
    """One row of the conformance table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    side: AcpSide
    status: AcpMethodStatus = AcpMethodStatus.NOT_IMPLEMENTED
    counterpart: str | None = Field(
        default=None,
        description="The UClone-X symbol this maps onto, or None where there is no counterpart.",
    )
    note: str = ""
    spec_section: str | None = Field(
        default=None, description="Section of docs/acp-protocol-spec.md that governs this row."
    )

    @model_validator(mode="after")
    def _reason_required_for_refusals(self) -> AcpMethod:
        """A refusal without a reason is indistinguishable from an oversight.

        §0 requires a reason for *Out of scope*, and §3 exists because *Not implementable*
        rows need their cause named. Enforcing it here means the registry cannot express the
        unexplained refusal at all, rather than relying on a reviewer noticing an empty cell.
        """
        refusals = (AcpMethodStatus.NOT_IMPLEMENTABLE, AcpMethodStatus.OUT_OF_SCOPE)
        if self.status in refusals and not self.note.strip():
            raise ValueError(
                f"ACP method {self.name!r} is {self.status.value} with no reason given"
            )
        return self


_AGENT_METHODS: tuple[AcpMethod, ...] = (
    AcpMethod(
        name="initialize",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="capability negotiation",
        note="Must report our real capabilities, not a fixed set.",
        spec_section="3.3",
    ),
    AcpMethod(
        name="new_session",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart=(
            "A per-session agent (agent_config_for_persona + compose_agent) "
            "+ BaseAgent.persist_session"
        ),
        note="Carries mcp_servers. Each session gets its own agent (#1454).",
        spec_section="5",
    ),
    AcpMethod(
        name="load_session",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="SessionStore.load + BaseAgent.hydrate_session",
        note="The saved history is restored into the session's own agent (#1454).",
    ),
    AcpMethod(
        name="list_sessions",
        side=AcpSide.AGENT,
        counterpart="SessionStore.list_session_ids",
        note="Ours returns ids only; ACP's response and its cursor pagination need more.",
        spec_section="3.4",
    ),
    AcpMethod(
        name="fork_session",
        side=AcpSide.AGENT,
        note="Needs a copy-with-new-id; SessionStore has no fork.",
    ),
    AcpMethod(
        name="resume_session",
        side=AcpSide.AGENT,
        counterpart="SessionStore.load",
        note="Distinct from load_session in ACP; the difference must be honoured, not collapsed.",
    ),
    AcpMethod(
        name="close_session",
        side=AcpSide.AGENT,
        note="Close is not delete. Mapping it to SessionStore.delete would lose the session.",
        spec_section="3.2",
    ),
    AcpMethod(
        name="set_session_mode",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="ACPServer._handle_set_session_mode",
    ),
    AcpMethod(
        name="set_config_option",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="ACPServer._handle_set_config_option",
        note="Boolean and select variants. Configuration reaches connectors through the UI settings panel.",
    ),
    AcpMethod(
        name="authenticate",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.OUT_OF_SCOPE,
        note="Out of scope for the local stdio head: the client is a local process. "
        "Adding a remote transport reopens this.",
        spec_section="3.5",
    ),
    AcpMethod(
        name="prompt",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="BaseAgent turn execution + BaseAgent.persist_session",
        note=("The core mapping. Every turn is saved; an unopened session id is refused (#1454)."),
    ),
    AcpMethod(
        name="cancel",
        side=AcpSide.AGENT,
        status=AcpMethodStatus.IMPLEMENTED,
        counterpart="ACPServer._handle_cancel",
        note="The mechanism exists on the sibling protocol; the gap is addressing and cooperation.",
        spec_section="3.1",
    ),
    AcpMethod(name="ext_method", side=AcpSide.AGENT, note="ACP's extension escape hatch."),
    AcpMethod(name="ext_notification", side=AcpSide.AGENT, note="ACP's extension escape hatch."),
    AcpMethod(name="on_connect", side=AcpSide.AGENT, note="SDK plumbing, not a protocol decision."),
)


_CLIENT_METHODS: tuple[AcpMethod, ...] = (
    AcpMethod(
        name="session_update",
        side=AcpSide.CLIENT,
        counterpart="AgentEvent to ACP update translation",
        note="The main translation surface. Blocked on #566, which changes what the turn "
        "loop emits; this shape must not be frozen ahead of it.",
        spec_section="4",
    ),
    AcpMethod(
        name="request_permission",
        side=AcpSide.CLIENT,
        counterpart="TOOL_APPROVAL_RESPONSE event",
        note="ACP makes permission a first-class request/response. Our nearest equivalent "
        "carries a 30s timeout, which is a decision ACP does not have and would need stating.",
        spec_section="4",
    ),
    AcpMethod(
        name="read_text_file",
        side=AcpSide.CLIENT,
        counterpart="tools/builtin/filesystem.py",
        note="Routing file I/O through the client is how an editor shows unsaved buffers. "
        "Optional, and a real behaviour change.",
        spec_section="4",
    ),
    AcpMethod(
        name="write_text_file",
        side=AcpSide.CLIENT,
        counterpart="tools/builtin/filesystem.py",
        note="Optional, and a real behaviour change.",
        spec_section="4",
    ),
    AcpMethod(
        name="create_terminal",
        side=AcpSide.CLIENT,
        counterpart="tools/builtin/shell.py",
        note="A client-hosted terminal runs outside our sandbox model entirely.",
        spec_section="4",
    ),
    AcpMethod(name="terminal_output", side=AcpSide.CLIENT, counterpart="tools/builtin/shell.py"),
    AcpMethod(name="release_terminal", side=AcpSide.CLIENT, counterpart="tools/builtin/shell.py"),
    AcpMethod(
        name="wait_for_terminal_exit", side=AcpSide.CLIENT, counterpart="tools/builtin/shell.py"
    ),
    AcpMethod(name="kill_terminal", side=AcpSide.CLIENT, counterpart="tools/builtin/shell.py"),
    AcpMethod(
        name="create_elicitation",
        side=AcpSide.CLIENT,
        note="Agent-initiated questions to the user. No counterpart.",
        spec_section="4",
    ),
    AcpMethod(
        name="complete_elicitation",
        side=AcpSide.CLIENT,
        note="Agent-initiated questions to the user. No counterpart.",
        spec_section="4",
    ),
    AcpMethod(name="ext_method", side=AcpSide.CLIENT, note="ACP's extension escape hatch."),
    AcpMethod(name="ext_notification", side=AcpSide.CLIENT, note="ACP's extension escape hatch."),
    AcpMethod(
        name="on_connect",
        side=AcpSide.CLIENT,
        note="SDK plumbing, not a protocol decision. Listed because §1 promises the table "
        "cannot silently omit a method, and an earlier version omitted exactly this one.",
    ),
)


class AcpMcpDescriptor(BaseModel):
    """How one ACP MCP-server descriptor shape maps onto `MCPConnectionConfig`.

    `new_session`, `load_session`, `fork_session` and `resume_session` all carry
    `mcp_servers`, so these arrive over the protocol from whatever client connected. §5
    records what was measured in #577 rather than what the field names suggest, because the
    field list looks compatible until you watch a value change on the way through.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: AcpMethodStatus
    note: str

    @model_validator(mode="after")
    def _reason_required_for_refusals(self) -> AcpMcpDescriptor:
        refusals = (AcpMethodStatus.NOT_IMPLEMENTABLE, AcpMethodStatus.OUT_OF_SCOPE)
        if self.status in refusals and not self.note.strip():
            raise ValueError(f"ACP descriptor {self.name!r} is {self.status.value} with no reason")
        return self


_MCP_DESCRIPTORS: tuple[AcpMcpDescriptor, ...] = (
    AcpMcpDescriptor(
        name="McpServerStdio",
        status=AcpMethodStatus.NOT_IMPLEMENTED,
        note="Maps onto MCPTransport's stdio member.",
    ),
    AcpMcpDescriptor(
        name="McpServerSse",
        status=AcpMethodStatus.NOT_IMPLEMENTED,
        note="Maps onto MCPTransport's SSE member.",
    ),
    AcpMcpDescriptor(
        name="McpServerHttp",
        status=AcpMethodStatus.NOT_IMPLEMENTED,
        note="MCP streamable HTTP. MCPTransport has no member for it (#579), and a "
        "descriptor of that shape is silently inferred as SSE rather than refused.",
    ),
    AcpMcpDescriptor(
        name="AcpMcpServer",
        status=AcpMethodStatus.NOT_IMPLEMENTABLE,
        note="Carries a serverId and no address, because the MCP server is reached back down "
        "the ACP connection to the client. There is no local process and no URL, and "
        "MCPConnectionConfig has no shape for it. Must be refused with a stated reason "
        "rather than silently dropped.",
    ),
)

# §5: client-supplied descriptors must not go through `MCPConfigFileLoader.parse_config_dict`,
# which expands `${VAR}` from the host environment — a client could name any host environment
# variable and receive its value in the argv of a process the same client chose. Stated here
# because the surface that lists descriptors is where an operator would otherwise assume the
# ordinary loader ran.
ACP_MCP_LOADER_WARNING = (
    "Client-supplied MCP descriptors must not be parsed by MCPConfigFileLoader."
    "parse_config_dict: it expands ${VAR} from the host environment, so a client could name "
    "any host environment variable and receive its value in a process argv."
)


def acp_mcp_descriptor_registry() -> tuple[AcpMcpDescriptor, ...]:
    """Every MCP-server descriptor shape ACP can hand us, and what we do with it."""
    return _MCP_DESCRIPTORS


def acp_method_registry() -> tuple[AcpMethod, ...]:
    """Every ACP method this repository has an opinion about, agent side then client side."""
    return _AGENT_METHODS + _CLIENT_METHODS


def implemented() -> tuple[AcpMethod, ...]:
    """The methods `initialize` may claim.

    Deliberately a function over the registry rather than a second list: a capability response
    assembled independently is the failure §3.3 describes.
    """
    return tuple(m for m in acp_method_registry() if m.status is AcpMethodStatus.IMPLEMENTED)


class AcpShellPresence(BaseModel):
    """Whether an ACP shell and its SDK are actually here.

    Measured rather than declared, so this turns true on its own when #649 lands instead of
    needing a literal edited in a second place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shell_module_present: bool
    sdk_installed: bool
    transport: str = ACP_TRANSPORT
    sdk_version_specified: str = ACP_SDK_VERSION
    reason: str

    @property
    def serving(self) -> bool:
        """True only when a shell could actually answer a client."""
        return self.shell_module_present and self.sdk_installed


def _module_present(name: str) -> bool:
    """Whether `name` is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A parent package that is itself missing raises rather than returning None.
        return False


def shell_presence() -> AcpShellPresence:
    """Report whether ACP is actually served here, and say why when it is not.

    P6: "no ACP shell is installed" and "a shell is running with no sessions" must not render
    as the same empty state. The reason string is what keeps them apart, and it names the
    missing half rather than reporting a bare False.
    """
    shell = _module_present("uclone_x.acp.server")
    sdk = _module_present("acp")

    if shell and sdk:
        reason = "An ACP shell module and the agent-client-protocol SDK are both present."
    elif not shell and not sdk:
        reason = (
            "No ACP shell is installed: uclone_x.acp.server does not exist and the "
            f"agent-client-protocol SDK ({ACP_SDK_VERSION}) is not a dependency. This is #649."
        )
    elif not shell:
        reason = (
            "The agent-client-protocol SDK is installed, but uclone_x.acp.server does not "
            "exist, so nothing answers a client."
        )
    else:
        reason = (
            "uclone_x.acp.server exists, but the agent-client-protocol SDK is not installed, "
            "so the shell cannot start."
        )

    return AcpShellPresence(shell_module_present=shell, sdk_installed=sdk, reason=reason)


class AcpSideCounts(BaseModel):
    """How many methods of one side sit in each state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int
    implemented: int
    not_implemented: int
    not_implementable: int
    out_of_scope: int


class AcpConformanceReport(BaseModel):
    """The whole picture, in the shape the UI renders and `initialize` will derive from.

    Typed rather than a bare dict because this is a wire contract: the surface reads every
    field, and a `dict[str, object]` pushes the checking onto whoever consumes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: str
    sdk_version_specified: str
    presence: AcpShellPresence
    serving: bool
    methods: tuple[AcpMethod, ...]
    mcp_descriptors: tuple[AcpMcpDescriptor, ...]
    mcp_loader_warning: str
    counts: dict[str, AcpSideCounts]


def conformance_summary() -> AcpConformanceReport:
    """Assemble the report from the registry, never from a parallel literal (§3.3)."""
    registry = acp_method_registry()
    presence = shell_presence()

    def _count(side: AcpSide, status: AcpMethodStatus) -> int:
        return sum(1 for m in registry if m.side is side and m.status is status)

    return AcpConformanceReport(
        transport=ACP_TRANSPORT,
        sdk_version_specified=ACP_SDK_VERSION,
        presence=presence,
        serving=presence.serving,
        methods=registry,
        mcp_descriptors=acp_mcp_descriptor_registry(),
        mcp_loader_warning=ACP_MCP_LOADER_WARNING,
        counts={
            side.value: AcpSideCounts(
                total=sum(1 for m in registry if m.side is side),
                implemented=_count(side, AcpMethodStatus.IMPLEMENTED),
                not_implemented=_count(side, AcpMethodStatus.NOT_IMPLEMENTED),
                not_implementable=_count(side, AcpMethodStatus.NOT_IMPLEMENTABLE),
                out_of_scope=_count(side, AcpMethodStatus.OUT_OF_SCOPE),
            )
            for side in AcpSide
        },
    )
