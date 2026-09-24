"""Unit tests validating models and protocols across all 10 UClone-X subsystems."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.a2a import AgentCard, TaskMessage, TaskResult, TaskStatus, WireProtocolType
from uclone_x.agent import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
    BaseAgentProtocol,
    FsScope,
    ModelTier,
    PersonaDefinition,
    SubagentInvocation,
    SubAgentSpec,
    SubAgentSupervisorProtocol,
    TurnResult,
)
from uclone_x.code_intel import (
    Diagnostic,
    DiagnosticSeverity,
    IndexFreshness,
    SymbolKind,
    SymbolLocation,
    SymbolLookup,
    SymbolNode,
)
from uclone_x.core import Provenance
from uclone_x.engine.protocols import EventBusProtocol, EventSubscriptionProtocol
from uclone_x.llm import (
    ChatMessage,
    FinishReason,
    LLMProviderProtocol,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenBudget,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.ontology import (
    EntitySchema,
    OntologyValidationResult,
    RelationSchema,
)
from uclone_x.sandbox import (
    ExecutionRequest,
    ExecutionResult,
    IsolationLevel,
)
from uclone_x.skills import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.telemetry import (
    MetricKind,
    MetricRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TraceRecorderProtocol,
)
from uclone_x.tools import (
    MCPConnectionConfig,
    MCPTransport,
    ToolContext,
    ToolProtocol,
    ToolResult,
)


def test_agent_models_and_immutability() -> None:
    llm_cfg = AgentLLMConfig(
        model_tier=ModelTier.PRO,
        model_name="claude-3-7-sonnet",
        temperature=0.2,
        max_tokens=4096,
        token_budget=TokenBudget(max_tokens=200_000),
    )
    config = AgentConfig(
        agent_id="agent-1",
        name="Tester",
        role="Lead Architect",
        description="System architect persona",
        system_prompt="Design architectures",
        llm_config=llm_cfg,
    )
    assert config.agent_id == "agent-1"
    assert config.role == "Lead Architect"
    assert config.llm_config.model_tier == ModelTier.PRO
    assert config.llm_config.model_name == "claude-3-7-sonnet"
    assert config.max_subagent_depth == 2

    # Immutability check
    with pytest.raises(ValidationError):
        config.name = "Mutated"  # type: ignore[misc]

    # Extra fields rejected
    with pytest.raises(ValidationError):
        AgentConfig(agent_id="1", name="T", system_prompt="S", extra_field="bad")  # type: ignore[call-arg]

    context = AgentContext(session_id="s1", agent_id="agent-1")
    assert context.current_state == AgentState.IDLE

    # PersonaDefinition test
    persona = PersonaDefinition(
        name="security_reviewer",
        role="Security Reviewer",
        description="Audits codebase for security vulnerabilities",
        system_prompt="Focus on AST, auth, and sandbox boundaries",
        allowed_tools=("grep_search", "view_file"),
        llm_config=AgentLLMConfig(model_tier=ModelTier.PRO),
        enable_write_tools=False,
        enable_subagent_tools=False,
    )
    assert persona.name == "security_reviewer"
    assert persona.enable_write_tools is False
    assert persona.llm_config.model_tier == ModelTier.PRO

    # SubagentInvocation test
    invocation = SubagentInvocation(
        type_name="security_reviewer",
        role="Security Auditor",
        prompt="Audit the auth module",
        fs_scope=FsScope.ISOLATED,
        llm_override=AgentLLMConfig(temperature=0.1),
    )
    assert invocation.type_name == "security_reviewer"
    assert invocation.fs_scope is FsScope.ISOLATED
    assert invocation.llm_override is not None and invocation.llm_override.temperature == 0.1

    spec = SubAgentSpec(
        name="Sub",
        role="Helper",
        system_prompt="Help",
        fs_scope=FsScope.ISOLATED,
        llm_config=AgentLLMConfig(model_tier=ModelTier.FAST),
    )
    assert spec.fs_scope is FsScope.ISOLATED
    assert spec.llm_config.model_tier == ModelTier.FAST

    tool_req = ToolCallRequest(id="call-0", name="search", arguments={"q": "CVE"})
    turn = TurnResult(
        turn_index=1,
        content="Done",
        tool_calls=(tool_req,),
        is_completed=True,
        provenance=Provenance.primary("anthropic", "claude-3-7-sonnet"),
    )
    assert turn.is_completed is True
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "search"


def test_tools_and_mcp_models() -> None:
    ctx = ToolContext(agent_id="a1", session_id="s1", workspace_root=Path("/srv/work"))
    assert ctx.isolation.level is IsolationLevel.WORKSPACE  # P3's decided default

    res = ToolResult(
        success=True,
        output={"data": 123},
        execution_time_ms=1.5,
        provenance=Provenance.primary("local_tool_runtime"),
    )
    assert res.success is True

    # Issue #37: ToolResult cannot have success=True with non-empty error
    with pytest.raises(ValidationError, match="cannot have success=True with a non-empty error"):
        ToolResult(success=True, error="unexpected failure", provenance=Provenance.primary("test"))

    mcp_cfg = MCPConnectionConfig(
        server_name="git_mcp",
        transport=MCPTransport.STDIO,
        command="npx",
        args=("-y", "mcp-git"),
        workspace_root=Path("/srv/work"),
    )
    assert mcp_cfg.server_name == "git_mcp"

    # Issue #580: When workspace_root is omitted for STDIO WorkspaceIsolation,
    # it defaults safely to Path.cwd() rather than raising or falling back to NoIsolation
    default_root_cfg = MCPConnectionConfig(server_name="test_no_root", command="python")
    assert default_root_cfg.isolation.level is IsolationLevel.WORKSPACE
    assert default_root_cfg.workspace_root == Path.cwd()

    # Issue #34: ContainerIsolation allow_network inconsistency raises ValidationError
    from uclone_x.sandbox import ContainerIsolation

    with pytest.raises(ValidationError, match="contradicts isolation.allow_network"):
        MCPConnectionConfig(
            server_name="test_conflict",
            isolation=ContainerIsolation(image="node", allow_network=True),
            allow_network=False,
        )

    container_cfg = MCPConnectionConfig(
        server_name="test_container",
        isolation=ContainerIsolation(
            image="node", allow_network=True, egress_allowlist=("api.github.com",)
        ),
        allow_network=True,
    )
    assert container_cfg.effective_allow_network is True
    assert container_cfg.effective_egress_allowlist == ("api.github.com",)


def test_llm_models() -> None:
    call = ToolCallRequest(id="call-1", name="fetch", arguments={"url": "http://example.com"})
    msg = ChatMessage(role=MessageRole.USER, content="Hello", tool_calls=(call,))
    assert msg.role == MessageRole.USER
    assert len(msg.tool_calls) == 1

    usage = TokenUsage(
        provider="anthropic",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )
    assert usage.provider == "anthropic"
    assert usage.total_tokens == 150

    # Issue #37: TokenUsage automatically derives total_tokens if omitted/zero
    derived_usage = TokenUsage(provider="ollama", input_tokens=40, output_tokens=60)
    assert derived_usage.total_tokens == 100

    # Issue #37: TokenUsage rejects mismatched total_tokens
    with pytest.raises(ValidationError, match="total_tokens"):
        TokenUsage(provider="ollama", input_tokens=40, output_tokens=60, total_tokens=999)

    resp = ModelResponse(
        finish_reason=FinishReason.STOP,
        content="Hi",
        usage=usage,
        model_name="claude-3-7-sonnet",
        provenance=Provenance.primary("anthropic", "claude-3-7-sonnet"),
    )
    assert resp.usage.total_tokens == 150

    chunk = StreamChunk(
        delta_content="Hello world", tool_calls=(call,), finish_reason=FinishReason.STOP
    )
    assert chunk.delta_content == "Hello world"
    assert len(chunk.tool_calls) == 1

    req = LLMRequest(
        model="gemini-2.5-pro",
        messages=(msg,),
        temperature=0.4,
        max_tokens=2048,
    )
    assert req.model == "gemini-2.5-pro"
    assert len(req.messages) == 1

    budget = TokenBudget(max_tokens=500_000)
    assert budget.max_tokens == 500_000


def test_sandbox_models() -> None:
    req = ExecutionRequest(
        command="python",
        args=("-c", "print(1)"),
        cwd=Path("/srv/work"),
        workspace_root=Path("/srv/work"),
    )
    assert req.isolation.level is IsolationLevel.WORKSPACE
    assert req.timeout_seconds == 60.0

    res = ExecutionResult(
        exit_code=0,
        stdout="1\n",
        duration_ms=10.0,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=Provenance.primary("workspace_runner"),
    )
    assert res.exit_code == 0


def test_a2a_models() -> None:
    card = AgentCard(name="Worker", description="Worker agent")
    assert card.version == "1.0.1"

    task = TaskMessage(
        task_id="t-1",
        session_id="s-1",
        sender_agent_id="lead",
        target_agent_id="worker",
    )
    assert task.task_id == "t-1"

    result = TaskResult(
        task_id="t-1",
        status=TaskStatus.COMPLETED,
        output_data={"res": True},
        provenance=Provenance.primary("worker"),
    )
    assert result.status == TaskStatus.COMPLETED

    assert WireProtocolType.LOCAL_IN_MEMORY == "local_in_memory"


def test_skills_and_ontology_models() -> None:
    manifest = SkillManifest(
        name="git-ops", description="Git operations", origin=SkillOrigin.SYNTHESIZED
    )
    assert manifest.version == "0.1.0"
    # Quarantined by default: a synthesized package is inert until something promotes it.
    assert manifest.status is SkillStatus.PENDING
    assert manifest.requested_isolation is None

    report = SkillAuditReport(
        skill_name="git-ops",
        is_safe=True,
        recommendation=AuditVerdict.APPROVE,
        risk_score=0.1,
    )
    assert report.recommendation is AuditVerdict.APPROVE
    assert AutoApprovalPolicy.SAFE_ONLY == "safe_only"

    # Fail-safe defaults: an unmeasured risk score is None, never a reassuring 0.0.
    unmeasured = SkillAuditReport(
        skill_name="git-ops", is_safe=False, recommendation=AuditVerdict.REQUIRE_HUMAN_REVIEW
    )
    assert unmeasured.risk_score is None

    # A report cannot omit its verdict and read as approval.
    with pytest.raises(ValidationError, match="recommendation"):
        SkillAuditReport.model_validate({"skill_name": "git-ops", "is_safe": True})

    # Issue #37: is_safe and recommendation cannot disagree
    with pytest.raises(ValidationError, match="cannot recommend APPROVE when is_safe is False"):
        SkillAuditReport(skill_name="git-ops", is_safe=False, recommendation=AuditVerdict.APPROVE)

    with pytest.raises(ValidationError, match="cannot recommend REJECT when is_safe is True"):
        SkillAuditReport(skill_name="git-ops", is_safe=True, recommendation=AuditVerdict.REJECT)

    entity = EntitySchema(name="Repository", description="Git repo")
    assert entity.name == "Repository"

    relation = RelationSchema(source_entity="Repository", predicate="has", target_entity="Commit")
    assert relation.is_directed is True

    val = OntologyValidationResult(is_valid=True)
    assert val.is_valid is True


def test_code_intel_and_telemetry_models() -> None:
    loc = SymbolLocation(
        file_path=Path("foo.py"), start_line=1, start_col=0, end_line=5, end_col=10
    )
    node = SymbolNode(name="process", kind=SymbolKind.FUNCTION, location=loc)
    assert node.kind == SymbolKind.FUNCTION

    diag = Diagnostic(file_path=Path("foo.py"), line=2, col=4, message="Type error")
    assert diag.severity is DiagnosticSeverity.ERROR

    # "no index" and "no definition" must not be the same answer (P6, code-intel spec 4).
    unavailable = SymbolLookup(freshness=IndexFreshness.UNAVAILABLE, reason="no LSP for .rs")
    empty_but_fresh = SymbolLookup(freshness=IndexFreshness.FRESH)
    assert unavailable.locations == empty_but_fresh.locations == ()
    assert unavailable.freshness is not empty_but_fresh.freshness

    span = SpanRecord(trace_id="tr-1", span_id="sp-1", name="agent.turn", kind=SpanKind.INTERNAL)
    assert span.name == "agent.turn"
    assert span.status is SpanStatus.UNSET

    metric = MetricRecord(name="turn_latency", kind=MetricKind.HISTOGRAM, value=42.5, unit="ms")
    assert metric.value == 42.5


def test_protocols_declare_the_members_they_are_named_for() -> None:
    """Replaces a test that asserted `isinstance(SomeProtocol, type)`.

    That is true of every class in Python, so it could not fail and did not detect that
    `EventBusProtocol` disagreed with `EventBus` in two of its four members
    (issue 2026-09-02-035). Structural conformance is now enforced statically in
    `tests/unit/test_protocol_conformance.py`; what is left here is the runtime fact
    that each protocol is a Protocol and is not empty.
    """
    protocols = [
        EventBusProtocol,
        EventSubscriptionProtocol,
        BaseAgentProtocol,
        SubAgentSupervisorProtocol,
        LLMProviderProtocol,
        TraceRecorderProtocol,
    ]

    for protocol in protocols:
        assert getattr(protocol, "_is_protocol", False), f"{protocol.__name__} is not a Protocol"
        members = {
            name
            for name in dir(protocol)
            if not name.startswith("_") or name in {"__aiter__", "__aenter__"}
        }
        assert members, f"{protocol.__name__} declares no members"

    # Non-runtime-checkable protocols deliberately refuse issubclass() and isinstance()
    # checks outright to prevent property side-effects. Static conformance is Pyright's job.
    check_subclass: Any = issubclass
    for proto in (BaseAgentProtocol, ToolProtocol, LLMProviderProtocol):
        with pytest.raises(TypeError, match="runtime_checkable"):
            check_subclass(dict, proto)

    # Runtime-checkable protocols like SubAgentSupervisorProtocol allow runtime checks
    assert check_subclass(dict, SubAgentSupervisorProtocol) is False
