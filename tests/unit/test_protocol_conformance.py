"""Static conformance bindings between every protocol and a concrete type.

Issue 2026-09-02-035: at commit d367613 the tree contained 30 protocols, one
implementation, and a test asserting `isinstance(SomeProtocol, type)` for six protocols —
which is true of every class in Python. `EventBusProtocol` disagreed with `EventBus` in two
of its four members and the gate stayed green.

The mechanism here is the declaration itself, not the assertions. Each `_x: SomeProtocol
= SomeImplementation()` is checked by `pyright --strict` during the gate, so a protocol
that stops describing its implementation fails the build. The stubs exist for the
subsystems that have no implementation yet: writing one proves the protocol is
implementable at all, and it breaks loudly when a protocol changes shape, which is the
signal a comment cannot give.

Runtime assertions are kept to the few things that are genuinely runtime facts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from uclone_x.a2a import (
    A2ADiscoveryProtocol,
    A2ADiscoveryService,
    A2AHttpTransport,
    A2AInMemoryTransport,
    A2ATransportProtocol,
    TaskResult,
    WireProtocolType,
)
from uclone_x.agent import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    PersonaDefinition,
    SubagentInvocation,
    SubAgentSpec,
    TurnResult,
)
from uclone_x.agent.protocols import BaseAgentProtocol, SubAgentSupervisorProtocol
from uclone_x.agent.session import SessionStore
from uclone_x.code_intel import (
    ASTParser,
    ASTParserProtocol,
    Diagnostic,
    LSPClientProtocol,
    SCIPIndexer,
    SCIPIndexerProtocol,
    SymbolGraph,
    SymbolGraphProtocol,
    SymbolLookup,
)
from uclone_x.core.capability import Capability
from uclone_x.core.client_driver import ClientDriverProtocol
from uclone_x.core.host import HostProtocol
from uclone_x.core.log_offset import LogOffsetAllocatorProtocol
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.workspace import WorkspaceProtocol
from uclone_x.engine.event_bus import (
    AgentEvent,
    BackpressurePolicy,
    EventBus,
    EventCallback,
    EventPriority,
    EventSource,
    EventSubscription,
)
from uclone_x.engine.protocols import (
    EventBusProtocol,
    EventSubscriptionProtocol,
    PublisherHandleProtocol,
    SchedulerProtocol,
    TimerServiceProtocol,
)
from uclone_x.llm import (
    AnthropicConnector,
    ContextCompactor,
    GeminiConnector,
    OllamaConnector,
    OpenAIConnector,
    TokenBudgetManager,
)
from uclone_x.llm.models import (
    FinishReason,
    ModelResponse,
    TokenUsage,
)
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
)
from uclone_x.log.file_allocator import FileLogOffsetAllocator
from uclone_x.ontology import (
    OntologyEngine,
    OntologyEngineProtocol,
    OntologyInducer,
    OntologyInducerProtocol,
    OntologyValidatorProtocol,
)
from uclone_x.sandbox import (
    ExecutionResult,
    IsolationLevel,
    PathValidator,
    PathValidatorProtocol,
    SandboxRunnerProtocol,
    WorkspaceSandboxRunner,
)
from uclone_x.skills import (
    Skill,
    SkillAuditor,
    SkillAuditorProtocol,
    SkillManifest,
    SkillOrigin,
    SkillProtocol,
    SkillRegistry,
    SkillRegistryProtocol,
    SkillSynthesizer,
    SkillSynthesizerProtocol,
)
from uclone_x.telemetry import (
    CompositeTelemetryExporter,
    InMemoryTelemetryExporter,
    LangfuseTelemetryExporter,
    MetricRecorderProtocol,
    MetricsCollector,
    OTLPTelemetryExporter,
    SpanStreamProtocol,
    TelemetryExporterProtocol,
    TelemetryTracer,
    TraceRecorderProtocol,
)
from uclone_x.tools import (
    BashRunTool,
    ComfyImageGenTool,
    DirectoryListTool,
    DuckDuckGoSearchProvider,
    FileEditTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
    LocalTool,
    MCPClient,
    MCPClientProtocol,
    MCPConnectionConfig,
    SearchProviderProtocol,
    ToolProtocol,
    ToolRegistry,
    ToolRegistryProtocol,
    ToolResult,
    WebFetchTool,
    WebSearchTool,
    create_default_registry,
)

# ======================================================================================
# 1. Engine — the only subsystem with a real implementation
# ======================================================================================

_bus: EventBusProtocol = EventBus()
_subscription: EventSubscriptionProtocol = EventSubscription(bus=EventBus(), topics={"*"})
_publisher_handle: PublisherHandleProtocol = _bus.register_publisher(
    sender_id="test_agent", source=EventSource.AGENT
)


class _Timer:
    def schedule_once(
        self,
        delay_seconds: float,
        callback: Callable[[], Awaitable[None]],
        timer_id: str | None = None,
    ) -> str:
        raise NotImplementedError

    def cancel(self, timer_id: str) -> bool:
        raise NotImplementedError


_T = TypeVar("_T")


class _Scheduler:
    async def submit(
        self,
        coro_fn: Callable[..., Awaitable[_T]],
        *args: object,
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs: object,
    ) -> _T:
        raise NotImplementedError


_timer: TimerServiceProtocol = _Timer()
_scheduler: SchedulerProtocol = _Scheduler()


# ======================================================================================
# 2. Agent
# ======================================================================================


_agent: BaseAgentProtocol = BaseAgent(config=AgentConfig(agent_id="test_agent", name="Test Agent"))


class _Supervisor:
    def define_persona(self, persona: PersonaDefinition) -> None:
        raise NotImplementedError

    def get_persona(self, name: str) -> PersonaDefinition | None:
        raise NotImplementedError

    async def spawn_subagent(self, spec: SubAgentSpec, parent_agent_id: str) -> str:
        raise NotImplementedError

    async def invoke_subagent(self, invocation: SubagentInvocation, parent_agent_id: str) -> str:
        raise NotImplementedError

    async def terminate_subagent(self, agent_id: str) -> bool:
        raise NotImplementedError

    def list_subagents(self, parent_agent_id: str | None = None) -> list[str]:
        raise NotImplementedError


_supervisor: SubAgentSupervisorProtocol = _Supervisor()


# ======================================================================================
# 3. A2A — concrete implementations
# ======================================================================================

_discovery_service = A2ADiscoveryService()
_discovery: A2ADiscoveryProtocol = _discovery_service
_in_memory_transport: A2ATransportProtocol = A2AInMemoryTransport()
_http_transport: A2ATransportProtocol = A2AHttpTransport()
_transport: A2ATransportProtocol = _in_memory_transport


# ======================================================================================
# 4. LLM — concrete implementations
# ======================================================================================

_provider: LLMProviderProtocol = OllamaConnector()
# The three keyed connectors now refuse construction without a credential rather than
# defaulting `api_key` to `""` and failing as a provider 401 on the first billed call
# (#385). This module only checks structural protocol conformance, so any non-blank key
# satisfies it; it must be passed explicitly because `OPENAI_API_KEY` and friends are
# unset in CI, which is exactly the configuration the refusal exists to catch.
_openai_provider: LLMProviderProtocol = OpenAIConnector(api_key="conformance-probe-key")
_anthropic_provider: LLMProviderProtocol = AnthropicConnector(api_key="conformance-probe-key")
_gemini_provider: LLMProviderProtocol = GeminiConnector(api_key="conformance-probe-key")
_budget: TokenBudgetManagerProtocol = TokenBudgetManager()
_compactor: ContextCompactorProtocol = ContextCompactor()


# ======================================================================================
# 5. Tools and MCP — concrete implementations
# ======================================================================================

_tool: ToolProtocol = LocalTool(name="test_tool", description="A test tool")
_file_read_tool: ToolProtocol = FileReadTool()
_file_write_tool: ToolProtocol = FileWriteTool()
_file_edit_tool: ToolProtocol = FileEditTool()
_file_search_tool: ToolProtocol = FileSearchTool()
_directory_list_tool: ToolProtocol = DirectoryListTool()
_bash_tool: ToolProtocol = BashRunTool()
_comfy_tool: ToolProtocol = ComfyImageGenTool()
_web_fetch_tool: ToolProtocol = WebFetchTool()
_web_search_tool: ToolProtocol = WebSearchTool()
_ddg_provider: SearchProviderProtocol = DuckDuckGoSearchProvider()
_tool_registry: ToolRegistryProtocol = ToolRegistry()
_default_registry: ToolRegistryProtocol = create_default_registry()
_mcp: MCPClientProtocol = MCPClient(
    config=MCPConnectionConfig(
        server_name="test_mcp", command="python", workspace_root=Path("/tmp")
    )
)


# ======================================================================================
# 6. Skills — concrete implementations
# ======================================================================================

_skill: SkillProtocol = Skill(
    manifest=SkillManifest(name="test_skill", description="desc", origin=SkillOrigin.HUMAN),
    instructions_markdown="# Markdown",
)
_skill_registry: SkillRegistryProtocol = SkillRegistry()
_auditor: SkillAuditorProtocol = SkillAuditor()


_synthesizer: SkillSynthesizerProtocol = SkillSynthesizer()


# ======================================================================================
# 7. Sandbox — concrete implementations
# ======================================================================================

_validator: PathValidatorProtocol = PathValidator()
_runner: SandboxRunnerProtocol = WorkspaceSandboxRunner()


# ======================================================================================
# 8. Ontology — concrete implementations
# ======================================================================================

_real_ontology_engine = OntologyEngine()
_ontology_validator: OntologyValidatorProtocol = _real_ontology_engine
_inducer: OntologyInducerProtocol = OntologyInducer(engine=_real_ontology_engine)
_ontology: OntologyEngineProtocol = _real_ontology_engine


# ======================================================================================
# 9. Telemetry — concrete implementations
# ======================================================================================

_real_tracer = TelemetryTracer()
_recorder: TraceRecorderProtocol = _real_tracer
_metrics: MetricRecorderProtocol = MetricsCollector()
_exporter: TelemetryExporterProtocol = InMemoryTelemetryExporter()
_otlp_exporter: TelemetryExporterProtocol = OTLPTelemetryExporter()
_langfuse_exporter: TelemetryExporterProtocol = LangfuseTelemetryExporter(
    public_key="pk_test", secret_key="sk_test"
)
_composite_exporter: TelemetryExporterProtocol = CompositeTelemetryExporter(
    [_exporter, _otlp_exporter, _langfuse_exporter]
)
_span_stream: SpanStreamProtocol = _real_tracer


# ======================================================================================
# 10. Code intelligence — concrete ASTParser, SymbolGraph, and SCIPIndexer implementations
# ======================================================================================

_parser: ASTParserProtocol = ASTParser()
_graph: SymbolGraphProtocol = SymbolGraph()
_scip_indexer: SCIPIndexerProtocol = SCIPIndexer()


class _LSPClient:
    async def initialize(self, workspace_root: Path) -> None:
        raise NotImplementedError

    async def get_diagnostics(self, file_path: Path) -> tuple[Diagnostic, ...]:
        raise NotImplementedError

    async def go_to_definition(self, file_path: Path, line: int, col: int) -> SymbolLookup:
        raise NotImplementedError


_lsp: LSPClientProtocol = _LSPClient()


# ======================================================================================
# Runtime assertions — only for things that are genuinely runtime facts
# ======================================================================================


def test_every_protocol_is_bound_to_a_concrete_type() -> None:
    """The static bindings above are checked by pyright --strict.

    This runtime test validates that each subsystem's concrete implementation conforms
    to the expected protocol contract without silent missing attributes.
    """
    assert _bus.backpressure_policy is BackpressurePolicy.ERROR
    assert _subscription.is_closed is False
    assert _runner.level is IsolationLevel.WORKSPACE
    assert _provider.provider_name == "ollama"
    assert isinstance(_budget, TokenBudgetManagerProtocol)
    assert isinstance(_publisher_handle, PublisherHandleProtocol)
    assert _publisher_handle.sender_id == "test_agent"
    assert _publisher_handle.source is EventSource.AGENT
    assert _tool.name == "test_tool"
    assert isinstance(_graph, SymbolGraphProtocol)
    assert isinstance(_supervisor, SubAgentSupervisorProtocol)


def test_the_real_event_bus_satisfies_its_protocol_at_runtime_too() -> None:
    """The static binding is the guarantee; this pins the one runtime check we rely on."""
    bus = EventBus(maxsize=4, backpressure_policy=BackpressurePolicy.ERROR)

    assert isinstance(bus, EventBusProtocol)
    assert bus.maxsize == 4
    assert bus.backpressure_policy is BackpressurePolicy.ERROR


def test_callback_type_matches_what_the_bus_accepts() -> None:
    """EventCallback allows a sync or async callable; the protocol must not narrow it."""

    def sync_cb(event: AgentEvent) -> None:
        return None

    async def async_cb(event: AgentEvent) -> None:
        return None

    callbacks: Sequence[EventCallback] = [sync_cb, async_cb]

    bus = EventBus()
    unsubs = [bus.subscribe_callback("test.*", cb) for cb in callbacks]
    assert len(unsubs) == 2
    assert callable(unsubs[0])
    assert callable(unsubs[1])


def test_provenance_is_reachable_from_every_result_envelope() -> None:
    """P6 requires it on each boundary-crossing envelope; this asserts the set."""
    prov = Provenance.primary("test")
    envelopes: list[Any] = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            content="x",
            usage=TokenUsage(provider="test"),
            provenance=prov,
        ),
        ToolResult(success=True, provenance=prov),
        ExecutionResult(exit_code=0, isolation_level=IsolationLevel.WORKSPACE, provenance=prov),
        TurnResult(turn_index=0, content="x", provenance=prov),
        TaskResult(task_id="t", provenance=prov),
    ]

    assert all(getattr(e, "provenance", None) is prov for e in envelopes)


def test_the_real_path_validator_and_runner_satisfy_protocols() -> None:
    """Assert PathValidator and WorkspaceSandboxRunner satisfy runtime and static contracts."""
    validator = PathValidator()
    assert isinstance(validator, PathValidatorProtocol)

    runner = WorkspaceSandboxRunner(validator=validator)
    assert runner.level is IsolationLevel.WORKSPACE


def test_the_real_skill_registry_satisfies_protocol_at_runtime() -> None:
    """SkillRegistryProtocol is @runtime_checkable; assert concrete SkillRegistry satisfies it."""
    registry = SkillRegistry()
    assert isinstance(registry, SkillRegistryProtocol)


def test_the_real_ontology_engine_and_inducer_satisfy_protocols() -> None:
    """Ontology protocols are @runtime_checkable; assert concrete implementations satisfy them."""
    engine = OntologyEngine()
    inducer = OntologyInducer(engine=engine)

    assert isinstance(engine, OntologyValidatorProtocol)
    assert isinstance(engine, OntologyEngineProtocol)
    assert isinstance(inducer, OntologyInducerProtocol)


def test_the_real_tool_registry_and_mcp_client_satisfy_protocols() -> None:
    """Assert ToolRegistry, LocalTool, and MCPClient satisfy runtime and static contracts."""
    registry = ToolRegistry()
    assert isinstance(registry, ToolRegistryProtocol)

    tool = LocalTool(name="echo", description="echo tool")
    registry.register(tool)
    assert registry.get("echo") is tool
    assert len(registry.list_tools()) == 1

    mcp_cfg = MCPConnectionConfig(
        server_name="test_server", command="python", workspace_root=Path("/tmp")
    )
    client = MCPClient(config=mcp_cfg)
    assert client.config == mcp_cfg

    assert _file_read_tool.name == "file_read"
    assert _file_write_tool.name == "file_write"
    assert _file_edit_tool.name == "file_edit"
    assert _file_search_tool.name == "file_search"
    assert _directory_list_tool.name == "directory_list"
    assert _bash_tool.name == "bash_run"
    assert _web_fetch_tool.name == "web_fetch"
    assert _web_search_tool.name == "web_search"
    assert isinstance(_ddg_provider, SearchProviderProtocol)
    assert len(_default_registry.list_tools()) == 22


def test_the_real_llm_subsystem_satisfies_protocols_at_runtime() -> None:
    """Assert LLM subsystem implementations satisfy runtime and static contracts."""
    budget = TokenBudgetManager()
    compactor = ContextCompactor()
    ollama = OllamaConnector()
    # Keyed connectors refuse construction without a credential (#385); see the note at
    # the module-level instances above.
    openai = OpenAIConnector(api_key="conformance-probe-key")
    anthropic = AnthropicConnector(api_key="conformance-probe-key")
    gemini = GeminiConnector(api_key="conformance-probe-key")

    assert isinstance(budget, TokenBudgetManagerProtocol)
    assert isinstance(compactor, ContextCompactorProtocol)
    assert ollama.provider_name == "ollama"
    assert openai.provider_name == "openai"
    assert anthropic.provider_name == "anthropic"
    assert gemini.provider_name == "gemini"


def test_the_real_telemetry_subsystem_satisfies_protocols_at_runtime() -> None:
    """Assert TelemetryTracer, MetricsCollector, and all TelemetryExporter implementations satisfy protocols."""
    tracer = TelemetryTracer()
    metrics = MetricsCollector()
    in_memory = InMemoryTelemetryExporter()
    otlp = OTLPTelemetryExporter()
    langfuse = LangfuseTelemetryExporter(public_key="pk_test", secret_key="sk_test")
    composite = CompositeTelemetryExporter([in_memory, otlp, langfuse])

    assert isinstance(tracer, TraceRecorderProtocol)
    assert isinstance(tracer, SpanStreamProtocol)
    assert isinstance(metrics, MetricRecorderProtocol)
    assert isinstance(in_memory, TelemetryExporterProtocol)
    assert isinstance(otlp, TelemetryExporterProtocol)
    assert isinstance(langfuse, TelemetryExporterProtocol)
    assert isinstance(composite, TelemetryExporterProtocol)


def test_the_real_a2a_subsystem_satisfies_protocols_at_runtime() -> None:
    """Assert A2A discovery and transport implementations satisfy runtime and static contracts."""
    discovery = A2ADiscoveryService()
    in_memory = A2AInMemoryTransport()
    http_transport = A2AHttpTransport()

    assert isinstance(discovery, A2ADiscoveryProtocol)
    assert isinstance(http_transport, A2ADiscoveryProtocol)
    assert in_memory.transport_type == WireProtocolType.LOCAL_IN_MEMORY
    assert http_transport.transport_type == WireProtocolType.REST_SSE


def test_the_real_code_intel_subsystem_satisfies_protocols_at_runtime() -> None:
    """Assert ASTParser, SymbolGraph, and SCIPIndexer satisfy runtime and static contracts."""
    parser = ASTParser()
    graph = SymbolGraph()
    indexer = SCIPIndexer()

    assert isinstance(parser, ASTParserProtocol)
    assert isinstance(graph, SymbolGraphProtocol)
    assert isinstance(indexer, SCIPIndexerProtocol)


# ---------------------------------------------------------------------------------------
# Kernel contracts (the core/shell architecture note §6)
#
# These are the contracts a shell satisfies and the kernel programs against. They are
# declared before anything consumes them, which is the point: C1-C6 record that the kernel
# names concrete classes today, and a protocol that arrives after its consumers is a
# protocol shaped to whatever was already written.
#
# `SessionStoreProtocol` binds to the real `SessionStore`, so the seam is proved against a
# working implementation rather than a stub. The other two have no implementation yet;
# their stubs prove the protocols are implementable and break loudly if either changes
# shape.
# ---------------------------------------------------------------------------------------


_session_store: SessionStoreProtocol = SessionStore()


class _Workspace:
    @property
    def root(self) -> Path:
        raise NotImplementedError

    def resolve(self, relative: Path | str) -> Path:
        raise NotImplementedError


_workspace: WorkspaceProtocol = _Workspace()


class _Host:
    @property
    def llm(self) -> LLMProviderProtocol:
        raise NotImplementedError

    @property
    def tools(self) -> ToolRegistryProtocol:
        raise NotImplementedError

    @property
    def sessions(self) -> SessionStoreProtocol:
        raise NotImplementedError

    @property
    def tracer(self) -> TraceRecorderProtocol:
        raise NotImplementedError

    @property
    def bus(self) -> EventBusProtocol:
        raise NotImplementedError

    @property
    def workspace(self) -> WorkspaceProtocol | None:
        raise NotImplementedError

    @property
    def sandbox(self) -> SandboxRunnerProtocol | None:
        raise NotImplementedError

    @property
    def isolation_floor(self) -> IsolationLevel | None:
        raise NotImplementedError

    @property
    def available_isolation(self) -> frozenset[IsolationLevel]:
        raise NotImplementedError

    @property
    def capabilities(self) -> frozenset[Capability]:
        raise NotImplementedError


_host: HostProtocol = _Host()


class _HeadlessHost:
    """A host with no user filesystem and no ability to execute anything.

    Annotated as `HostProtocol` below, which is the point: an earlier version of this file
    declared a local class with three `None` attributes, asserted they were `None`, and
    advertised that as covering the optionality. It was a tautology — the class was never a
    `HostProtocol`, so the test would have passed with the protocol deleted or with
    `workspace` retyped as required.
    """

    @property
    def llm(self) -> LLMProviderProtocol:
        raise NotImplementedError

    @property
    def tools(self) -> ToolRegistryProtocol:
        raise NotImplementedError

    @property
    def sessions(self) -> SessionStoreProtocol:
        raise NotImplementedError

    @property
    def tracer(self) -> TraceRecorderProtocol:
        raise NotImplementedError

    @property
    def bus(self) -> EventBusProtocol:
        raise NotImplementedError

    @property
    def workspace(self) -> None:
        return None

    @property
    def sandbox(self) -> None:
        return None

    @property
    def isolation_floor(self) -> None:
        return None

    @property
    def available_isolation(self) -> frozenset[IsolationLevel]:
        return frozenset()

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset()


# The binding is the assertion: a host that returns `None` from all three optional members
# still satisfies the contract, so "this process has no filesystem and cannot execute" is
# expressible rather than inferred. Retyping any of them as required fails the gate here.
_headless_host: HostProtocol = _HeadlessHost()


def test_a_host_may_state_that_it_has_no_workspace_and_no_execution() -> None:
    """The optional members are optional in fact, not only in annotation.

    C3 is what this exists against: `Path.cwd()` hands a worker its own directory as the
    agent's workspace — a real path, the wrong answer, and nothing to say it was assumed.
    """
    host: HostProtocol = _HeadlessHost()
    assert host.workspace is None
    assert host.sandbox is None
    # A `None` floor is stricter than any level rather than a new one: there is no execution
    # at all, which `IsolationLevel.NONE` — still execution, just unisolated — cannot express.
    assert host.isolation_floor is None
    assert host.available_isolation == frozenset()


def test_capability_names_are_the_two_axes_kept_separate() -> None:
    """Workspace access and process execution are separate capabilities.

    Review finding 2026-09-02-019 recorded the vocabulary collision that came of conflating
    them — the Markdown findings register that held it is retired, and the snapshot
    survives as an eval fixture. A host may read and write files and never execute, or
    execute against a scratch volume with no user files, so neither implies the other.
    """
    names = {c.value for c in Capability}
    assert {"workspace.read", "workspace.write"} <= names
    assert "process.exec" in names
    # Distinct members, not distinct strings: an earlier version compared three literals to
    # each other, which is true of the literals and says nothing about the enum.
    assert (
        len({Capability.WORKSPACE_READ, Capability.WORKSPACE_WRITE, Capability.PROCESS_EXEC}) == 3
    )


# The allocator binds to the protocol, so a method missing from the *implementation* fails the
# gate. The direction matters and this comment used to have it backwards: an assignment checks
# that the implementation satisfies the protocol, so removing a method from
# `LogOffsetAllocatorProtocol` passes cleanly while renaming one on `FileLogOffsetAllocator`
# raises `reportAssignmentType`. Verified both ways. A protocol that loses a method is not
# caught here, and review finding 2026-09-02-035, "protocols unbound to implementations",
# is about the implementation direction, which this does catch. (The Markdown findings
# register that held it is retired; the snapshot survives as an eval fixture.)
#
# Deliberately under `TYPE_CHECKING` rather than a live module-level assignment: the
# constructor creates its root directory, so binding at import time made every test run leak a
# `mkdtemp()` directory nothing removed. Pyright evaluates this branch and rejects a missing
# method exactly as before; at runtime it never executes, so it creates nothing.
if TYPE_CHECKING:
    _log_offset_allocator: LogOffsetAllocatorProtocol = FileLogOffsetAllocator(root=Path())


class _ClientDriver:
    async def initialize(self) -> None: ...
    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]: ...
    async def cancel(self, reason: str) -> None: ...


_client_driver: ClientDriverProtocol = _ClientDriver()
