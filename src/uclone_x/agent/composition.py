"""Agent composition root (Issue #538)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks import BaseHook, HookRunner
from uclone_x.agent.models import AgentConfig, AgentContext, PersonaDefinition
from uclone_x.agent.persona_store import PersonaStoreProtocol
from uclone_x.agent.planner import PlanGenerator
from uclone_x.core.capability import Capability
from uclone_x.core.host import HostProtocol
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.workspace import WorkspaceProtocol
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
)
from uclone_x.llm.router import SemanticModelRouter
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.sandbox.models import AVAILABLE_ISOLATION_LEVELS, IsolationLevel
from uclone_x.sandbox.protocols import SandboxRunnerProtocol
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.telemetry.protocols import TracerProtocol
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.tool_scoper import ToolScoperProtocol

if TYPE_CHECKING:
    from uclone_x.a2a.protocols import A2ATransportProtocol


class MissingCapabilityError(Exception):
    """Raised when a required capability is missing during agent composition.

    `missing` names each capability, so a head can say which one in its own words rather
    than parse this message: a missing `llm` is the user's to fix (choose a model), and
    every other one is ours (#1446).
    """

    def __init__(self, message: str, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


@dataclass
class HostDependencies:
    """Dependencies provided by the host environment for agent execution.

    Required capabilities (bus, llm, tools, tracer, store) must be provided.
    Optional capabilities remain optional.
    """

    bus: EventBusProtocol | None = None
    llm: LLMProviderProtocol | None = None
    tools: ToolRegistryProtocol | None = None
    tracer: TracerProtocol | None = None
    store: SessionStoreProtocol | None = None

    # Optional capabilities
    ontology: OntologyEngineProtocol | None = None
    skills: SkillRegistryProtocol | None = None
    # Cross-session memory. Optional on the dataclass and genuinely optional in fact:
    # `BaseAgent` registers `record_memory_fact` / `query_memory_facts` only when it is
    # given a store, so an agent composed without one has no memory tools -- and since
    # #1098 cannot borrow another agent's out of a shared registry either. Adding the
    # field is what made wiring possible at all; each head still has to do it, and a head
    # seating several agents must give each its own store rather than one they share.
    memory: CrossSessionMemory | None = None
    budget: TokenBudgetManagerProtocol | None = None
    compactor: ContextCompactorProtocol | None = None
    hooks: Sequence[BaseHook] | None = None
    hook_runner: HookRunner | None = None
    semantic_router: SemanticModelRouter | None = None
    tool_scoper: ToolScoperProtocol | None = None
    plan_generator: PlanGenerator | None = None
    persona: str | None = None
    persona_name: str | None = None
    persona_store: PersonaStoreProtocol | None = None
    #: Persona definitions registered on the agent before its session is seeded. For a
    #: definition that is not in the registry the agent reads, such as one a room was
    #: handed, so the anchor is composed with it in force.
    persona_definitions: tuple[PersonaDefinition, ...] = ()

    workspace: WorkspaceProtocol | None = None
    sandbox: SandboxRunnerProtocol | None = None
    isolation_floor: IsolationLevel | None = IsolationLevel.WORKSPACE
    available_isolation: frozenset[IsolationLevel] = AVAILABLE_ISOLATION_LEVELS
    #: How this agent's `a2a_call` reaches peer personas (#1558), or `None` when it may
    #: call no one. An agent built to answer a peer call is given `None`, which is what
    #: holds such calls to one level deep; a sub-agent is not given it either.
    a2a_transport: A2ATransportProtocol | None = None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset()


# `HostDependencies` fields a sub-agent is *not* given (#1449). Every other field is
# forwarded by `BaseAgent._subagent_host_fields`; a unit test holds the two sets to exactly
# the dataclass's fields, so a new field has to be placed in one of them. A child has no
# persona of its own and no cross-session memory, and gets a hook runner of its own seeded
# with its parent's hooks (so its hook events carry its own sender id).
SUBAGENT_EXCLUDED_HOST_FIELDS: frozenset[str] = frozenset(
    {
        "memory",
        "hook_runner",
        "persona",
        "persona_name",
        "persona_store",
        "persona_definitions",
        "a2a_transport",
    }
)


def compose_agent(
    config: AgentConfig,
    host: HostDependencies,
    context: AgentContext | None = None,
) -> BaseAgent:
    """Compose a BaseAgent instance, validating required host capabilities."""
    missing: list[str] = []
    if host.bus is None:
        missing.append("bus")
    if host.llm is None:
        missing.append("llm")
    if host.tools is None:
        missing.append("tools")
    if host.tracer is None:
        missing.append("tracer")
    if host.store is None:
        missing.append("store")

    if missing:
        missing_str = ", ".join(missing)
        raise MissingCapabilityError(
            f"Missing required capabilities for agent composition: {missing_str}",
            missing=tuple(missing),
        )

    return build_agent(config, host, context)


def build_agent(
    config: AgentConfig,
    host: HostDependencies,
    context: AgentContext | None = None,
) -> BaseAgent:
    """Construct a BaseAgent from `host`, without the required-capability check.

    `compose_agent` is how a new agent is made. This is its construction half, for an agent
    derived from one that already exists -- `BaseAgent.spawn_subagent` (#1449). A child's
    host is its parent's, so the check has nothing to add: what the parent lacks, the child
    lacks too, and refusing there would make a working parent unable to delegate. There is
    one mapping from `HostDependencies` to `BaseAgent`, and it is this. Which fields a child
    gets is decided separately, field by field, in `BaseAgent._subagent_host_fields` and
    `SUBAGENT_EXCLUDED_HOST_FIELDS`; a new field is not forwarded until someone adds it there.
    """
    return BaseAgent(
        config=config,
        bus=host.bus,
        llm=host.llm,
        tools=host.tools,
        context=context,
        tracer=host.tracer,
        store=host.store,
        ontology=host.ontology,
        skills=host.skills,
        memory=host.memory,
        compactor=host.compactor,
        budget=host.budget,
        hooks=host.hooks,
        hook_runner=host.hook_runner,
        semantic_router=host.semantic_router,
        tool_scoper=host.tool_scoper,
        plan_generator=host.plan_generator,
        persona=host.persona,
        persona_name=host.persona_name,
        personas=host.persona_definitions,
        host=cast("HostProtocol", host),
        a2a_transport=host.a2a_transport,
    )
