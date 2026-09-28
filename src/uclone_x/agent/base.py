"""Concrete BaseAgent implementing 6-stage reactive state machine (FR-1, P1, P4, P6, P8)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from uclone_x.agent.artist_skill_router import (
    CASE_SKILL_PERSONAS,
    case_skill_section,
    extract_request_facts,
    grounded_facts,
    is_follow_up,
    route_first_turn,
    route_follow_up,
)
from uclone_x.agent.compaction_driver import CompactionDriver, CompactionScope
from uclone_x.agent.hooks import (
    BaseHook,
    HookRunner,
)
from uclone_x.agent.image_set_planner import (
    IMAGE_SET_PERSONAS,
    detect_image_set,
    image_set_note,
    plan_image_set,
)
from uclone_x.agent.models import (
    BASE_MEMORY_TOOLS,
    AgentConfig,
    AgentContext,
    AgentState,
    PersonaDefinition,
    PlanState,
    PlanStep,
    ToolExecutionRecord,
    TurnResult,
)
from uclone_x.agent.planner import PlanGenerator
from uclone_x.agent.prompt_assembler import (
    AnchorWriter,
    PromptAssembler,
    PromptScope,
    compose_identity_prompt,
    composed_seat_framing,
)
from uclone_x.agent.prompts import adapt_system_prompt
from uclone_x.agent.protocols import BaseAgentProtocol, TurnLifecycleHookProtocol
from uclone_x.agent.request_record import (
    RequestLayers,
    assemble_request_messages,
)
from uclone_x.agent.session import (
    CompactionResult,
    SessionState,
)
from uclone_x.agent.session_lifecycle import (
    SessionLifecycle,
    SessionScope,
    _LiveSession,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.agent.tool_execution import ExecutionScope, ToolCallExecutor
from uclone_x.agent.tool_invoker import ToolInvoker, ToolScope
from uclone_x.agent.turn_executor import StreamProgress, TurnExecutor, TurnScope, named_model
from uclone_x.core.capability import Capability
from uclone_x.core.host import HostProtocol
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import (
    require_provenance,
)
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import (
    STEP_REPLY_RESERVE_TOKENS,
    artifacts_dir_for,
    ingest_tool_text,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventSource,
    EventType,
)
from uclone_x.engine.protocols import (
    EventBusProtocol,
    EventSubscriptionProtocol,
    PublisherHandleProtocol,
)
from uclone_x.errors import (
    InvalidStateTransitionError,
)
from uclone_x.llm.context_window import (
    SERVED_WINDOW_PROVIDERS,
    OllamaContextWindows,
    compaction_window,
)
from uclone_x.llm.models import (
    BudgetDecision,
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenBudget,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
)
from uclone_x.llm.router import SemanticModelRouter
from uclone_x.memory import (
    CrossSessionMemory,
    QueryMemoryFactsTool,
    ReadOnlyMemory,
    RecordMemoryFactTool,
    RetractMemoryFactTool,
)
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.sandbox.models import (
    AVAILABLE_ISOLATION_LEVELS,
    IsolationLevel,
    IsolationPolicy,
    WorkspaceIsolation,
    effective_isolation_level,
)
from uclone_x.skills.models import SkillStatus
from uclone_x.skills.protocols import SkillProtocol, SkillRegistryProtocol
from uclone_x.telemetry.protocols import TracerProtocol
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import (
    tool_spawns_subagents,
    tool_writes_files,
)
from uclone_x.tools.builtin.media_registry import ImageModelSource, ModelProfile
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.builtin.skill_proposer import ProposeSkillTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_binder import (
    ToolBinder,
)

if TYPE_CHECKING:
    from uclone_x.a2a.protocols import A2ATransportProtocol

logger = logging.getLogger(__name__)

# Upper bound on failures retained by `BaseAgent.processing_errors`.
_MAX_RECORDED_ERRORS = 100


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


VALID_TRANSITIONS: Mapping[AgentState, frozenset[AgentState]] = {
    AgentState.IDLE: frozenset({AgentState.INGESTING, AgentState.TERMINATED, AgentState.ERROR}),
    AgentState.INGESTING: frozenset(
        {AgentState.REASONING, AgentState.IDLE, AgentState.TERMINATED, AgentState.ERROR}
    ),
    AgentState.REASONING: frozenset(
        {
            AgentState.CALLING_TOOL,
            AgentState.EMITTING_RESPONSE,
            AgentState.AWAITING_INPUT,
            AgentState.IDLE,
            AgentState.ERROR,
        }
    ),
    AgentState.CALLING_TOOL: frozenset(
        {
            AgentState.AWAITING_INPUT,
            AgentState.INGESTING,
            AgentState.REASONING,
            AgentState.EMITTING_RESPONSE,
            AgentState.ERROR,
            AgentState.IDLE,
        }
    ),
    AgentState.AWAITING_INPUT: frozenset(
        {
            AgentState.INGESTING,
            AgentState.REASONING,
            AgentState.ERROR,
            AgentState.TERMINATED,
            AgentState.IDLE,
        }
    ),
    AgentState.EMITTING_RESPONSE: frozenset(
        {AgentState.IDLE, AgentState.INGESTING, AgentState.TERMINATED, AgentState.ERROR}
    ),
    AgentState.ERROR: frozenset({AgentState.IDLE, AgentState.TERMINATED}),
    AgentState.TERMINATED: frozenset(),
}


#: Tool classes whose *instance* is bound to one agent's own state, so a registry hit for
#: one of these is another agent's instance rather than a shared implementation. Resolution
#: and advertisement both refuse them for an agent that did not register its own.
#:
#: Matched by type and not by name, because the two answer differently for the case that
#: matters: a third-party or MCP tool registered under the name `query_memory_facts` is an
#: ordinary shared implementation and must resolve for every agent. Refusing it on its name
#: would report a registered, working tool as missing -- the same lie as the one this guard
#: exists to stop, pointed the other way (P6).
AGENT_BOUND_TOOL_TYPES: Final = (
    RecordMemoryFactTool,
    RetractMemoryFactTool,
    QueryMemoryFactsTool,
    LoadSkillTool,
)


class _ParentSessionBudget:
    """A sub-agent's view of its parent's budget: every booking lands on the parent's session.

    The ledger is keyed by session id and a child has a session of its own, so handing it
    the parent's manager unchanged would open a fresh session budget with the default
    ceiling -- the child's spend recorded, but never enforced against what the parent was
    allowed (P4, #1449). Rewriting the key is what makes the parent's ceiling the child's.
    """

    def __init__(self, parent: TokenBudgetManagerProtocol, parent_session_id: str) -> None:
        self._parent = parent
        self._session_id = parent_session_id

    def check_budget(self, session_id: str, provider: str | None = None) -> BudgetDecision:
        return self._parent.check_budget(self._session_id, provider=provider)

    def record_usage(self, session_id: str, usage: TokenUsage) -> None:
        self._parent.record_usage(self._session_id, usage)

    def enforce_budget(self, session_id: str, provider: str | None = None) -> None:
        self._parent.enforce_budget(self._session_id, provider=provider)

    def get_budget(self, session_id: str) -> TokenBudget | None:
        return self._parent.get_budget(self._session_id)

    def record_compaction(
        self,
        reason: str,
        original_tokens: int,
        compacted_tokens: int,
        kept_turns: int = 4,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        # A child's compaction of its own context lands in the parent session's history,
        # so the parent's summary counts compactions across the whole delegation tree.
        return self._parent.record_compaction(
            reason, original_tokens, compacted_tokens, kept_turns, session_id=self._session_id
        )

    def get_summary(self, session_id: str | None = None) -> dict[str, Any]:
        return self._parent.get_summary(self._session_id if session_id is not None else None)


class BaseAgent(BaseAgentProtocol):
    """Concrete reactive state machine agent with EventBus and LLM integration."""

    def define_persona(self, persona: PersonaDefinition) -> None:
        """Register a reusable dynamic persona definition, visible to this agent only.

        `_persona_store` is an *instance* attribute, assigned in `__init__`. It was a class
        attribute, and because `self._persona_store[...] = ...` mutates the one dict on the
        class body, every persona any agent defined was visible to every other agent in the
        process -- sub-agents in isolated contexts included -- and nothing cleared it between
        sessions. A registration is scoped to its agent; a persona meant for everyone belongs
        in a YAML file that `PersonaRegistry` discovers.

        A registration can change what the name in force resolves to without any
        assignment: a name set before its definition existed, or a definition replacing
        the one that was in force. The prompt follows that on its own, because
        `effective_system_prompt` re-resolves on every read. The tool scope is resolved
        once and stored, so it has to be recomputed here too. Otherwise the agent speaks as
        the persona while it can still call tools the persona withholds (#1153). The
        recomputation is the one the setters run, so the operator's list still wins, and
        registering a name that is not in force leaves the scope as it was.
        """
        self._persona_definition_commit(persona)()

    def _persona_definition_commit(self, persona: PersonaDefinition) -> Callable[[], None]:
        """Work out what registering `persona` changes, and return the step that applies it.

        Everything that can raise -- resolving the persona in force, reading its tools,
        copying the config -- runs here, against a candidate copy of the store, before any
        of the agent's state moves. The returned step only assigns. So a registration that
        fails leaves the agent wholly on the definition it had: not the new prompt (which
        `effective_system_prompt` reads from the store) under the old tool scope (#1904).
        """
        store = {**self._persona_store, persona.name: persona}
        config = self._persona_scoped_config(store)

        def commit() -> None:
            self._persona_store[persona.name] = persona
            self._config = config  # the tool scope, recomputed for the registration

        return commit

    def stage_persona_edit(self, persona: PersonaDefinition) -> None:
        """Hold a saved persona edit until this agent's next turn starts (#1899).

        A live seat takes a saved edit at a turn boundary, never mid-turn: `define_persona`
        changes the prompt and the tool scope, and a turn that is running -- or one that
        compaction is resolving the seat beside -- must finish under the persona it began
        with. The turn applies the edit under its turn lock and opens a new epoch for it,
        so the session log shows where the identity changed. A later edit replaces an
        earlier one that has not been taken yet.
        """
        self._staged_persona = persona

    def _take_staged_persona(self) -> Callable[[], None] | None:
        """Take the staged edit, if any; the step that applies it, or `None` if it changes nothing.

        Called only at a turn's start, under the turn lock (`TurnExecutor.execute_turn`).
        Nothing is applied here: the caller opens the edit's epoch first and then runs the
        returned step, which only assigns, so a raise anywhere before it leaves the seat on
        its old definition with no epoch opened (#1904). The edit is taken off the stage
        either way; one that fails to apply is dropped with its failed turn rather than
        failing every turn after it.
        """
        staged, self._staged_persona = self._staged_persona, None
        if staged is None or self._persona_store.get(staged.name) == staged:
            return None
        return self._persona_definition_commit(staged)

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Resolve a persona: this agent's own registrations first, then the registry.

        There is no third branch. `scout` and `critic` were returned from module constants
        ahead of the store, so a persona registered under either name was silently ignored
        rather than overriding or being refused; both now ship as YAML that the registry
        seeds, so the constants have no resolution role left.
        """
        return self._persona_in(self._persona_store, name)

    def _persona_in(
        self, store: Mapping[str, PersonaDefinition], name: str
    ) -> PersonaDefinition | None:
        """`get_persona`, resolved against `store` in place of this agent's own registrations."""
        if name in store:
            return store[name]
        from uclone_x.agent.persona_registry import get_default_persona_registry

        return get_default_persona_registry(self._workspace_root_hint).get_persona(name)

    def __init__(
        self,
        config: AgentConfig,
        bus: EventBusProtocol | None = None,
        llm: LLMProviderProtocol | None = None,
        tools: ToolRegistryProtocol | None = None,
        context: AgentContext | None = None,
        ontology: OntologyEngineProtocol | None = None,
        tracer: TracerProtocol | None = None,
        store: SessionStoreProtocol | None = None,
        compactor: ContextCompactorProtocol | None = None,
        budget: TokenBudgetManagerProtocol | None = None,
        hooks: Sequence[BaseHook] | None = None,
        hook_runner: HookRunner | None = None,
        semantic_router: SemanticModelRouter | None = None,
        tool_binder: ToolBinder | None = None,
        plan_generator: PlanGenerator | None = None,
        persona: str | None = None,
        persona_name: str | None = None,
        skills: SkillRegistryProtocol | None = None,
        host: HostProtocol | None = None,
        memory: CrossSessionMemory | None = None,
        personas: Sequence[PersonaDefinition] = (),
        a2a_transport: A2ATransportProtocol | None = None,
        approvals_answered: bool = True,
        lifecycle_hooks: Sequence[TurnLifecycleHookProtocol] = (),
    ) -> None:
        self._a2a_transport = a2a_transport
        # Domain behaviour added to every turn by whoever composed this agent (#1732).
        self._lifecycle_hooks: tuple[TurnLifecycleHookProtocol, ...] = tuple(lifecycle_hooks)
        # Whether anyone in this app answers an approval request during a turn. When not,
        # a call that needs a person is refused at once instead of after the timeout.
        self._approvals_answered = approvals_answered
        if host is None:

            class _FallbackHost:
                workspace = None
                sandbox = None
                isolation_floor = IsolationLevel.WORKSPACE
                available_isolation = AVAILABLE_ISOLATION_LEVELS

                @property
                def capabilities(self) -> frozenset[Capability]:
                    return frozenset()

            self._host: HostProtocol = cast(HostProtocol, _FallbackHost())
        else:
            self._host: HostProtocol = host

        self._config = config
        self._persona: str | None = (
            persona
            if persona is not None
            else (persona_name if persona_name is not None else getattr(config, "persona", None))
        )
        # Registered before the session is seeded, so the first anchor is composed from
        # them. A `define_persona` after construction leaves an anchor seeded without
        # them; the turn then re-resolves the prompt, and the record says something
        # other than what was sent.
        self._persona_store: dict[str, PersonaDefinition] = {p.name: p for p in personas}
        self._staged_persona: PersonaDefinition | None = None
        self._bus = bus
        self._llm = llm
        self._tools = tools
        self._skills = skills
        self._loaded_skills: set[str] = set()
        # Tools bound to *this* agent's own state, consulted by `ToolInvoker.resolve` before
        # the registry. The UI gives all four heads one `ToolRegistry`, so a tool registered
        # there and resolved from there is whichever agent's instance was composed first.
        # The registry copy still exists for *advertisement*: the schema and description
        # are agent-independent, and registering there is what keeps a persona's
        # `allowed_tools` whitelist able to scope them out (#1097).
        agent_local_tools: dict[str, ToolProtocol] = {}
        if self._skills is not None:
            if self._tools is None:
                self._tools = ToolRegistry()
            # `load_skill` is bound twice over: to `self._skills`, which decides which
            # skills are approved for *this* agent (P9), and to `self._record_loaded_skill`,
            # which is this agent's `_loaded_skills` set. Resolved from the shared registry,
            # the second agent got the first one's: it loaded from the first agent's
            # approval list, and the name landed in the first agent's `_loaded_skills` --
            # so its own prompt and telemetry never showed the skill it had just loaded,
            # its `reset_session` cleared nothing, and the other agent's reset cleared it.
            # The image domain skills are split by prompt family, and `load_skill` picks
            # the section for the model `generate_image` would use now. The registered
            # image tool is reached through the kernel-side `ImageModelSource` contract,
            # so this module does not import the image adapter; the same tool is given
            # the skill store its description lists the domains from (design §3.1-3.2).
            image_source = self._tools.get("generate_image")
            profile_provider: Callable[[], ModelProfile] | None = None
            if isinstance(image_source, ImageModelSource):
                image_source.bind_skill_registry(self._skills)
                profile_provider = image_source.active_profile
            skill_tool = LoadSkillTool(
                self._skills,
                on_load=self._record_loaded_skill,
                profile_provider=profile_provider,
                tool_scope=lambda: self._config.allowed_tools,
            )
            agent_local_tools[skill_tool.name] = skill_tool
            if self._tools.get(skill_tool.name) is None:
                self._tools.register(skill_tool)
            # `propose_skill` writes a pending proposal into the store's `.pending/` area,
            # which a person approves in Settings (#1827). Only a file-system store has
            # such a folder. Not a base tool: a clone with an `allowed_tools` list has it
            # only by naming it.
            store_root = getattr(self._skills, "store_root", None)
            if isinstance(store_root, Path):
                propose_tool = ProposeSkillTool(store_root)
                agent_local_tools[propose_tool.name] = propose_tool
                if self._tools.get(propose_tool.name) is None:
                    self._tools.register(propose_tool)
        self._ontology = ontology
        self._memory = memory
        # Memory tools are bound to *this* agent's store, so unlike every other tool they
        # cannot be shared: registering them in the shared registry and resolving them from
        # there would bind every later agent's tools to whichever agent was composed first
        # -- the critic would record into the champion's file.
        if self._memory is not None:
            if self._tools is None:
                self._tools = ToolRegistry()
            for memory_tool in (
                RecordMemoryFactTool(self._memory),
                RetractMemoryFactTool(self._memory),
                QueryMemoryFactsTool(self._memory),
            ):
                agent_local_tools[memory_tool.name] = memory_tool
                if self._tools.get(memory_tool.name) is None:
                    self._tools.register(memory_tool)
        self._store = store
        self._injected_compactor = compactor
        self._budget = budget
        # Agent steps taken in the current run — cleared by every externally initiated
        # turn. Distinct from `_turn_counter`, the lifetime count the session persists.
        self._run_steps = 0
        self._steps_deducted = False
        # The conversation and open story of the running turn (#1555), set by
        # `execute_turn` and given to every tool call's `ToolContext`.
        self._turn_room_id: str | None = None
        self._turn_caller_turn_id: str | None = None
        self._turn_story_id: str | None = None
        self._pending_durable_events: list[dict[str, Any]] = []
        self._semantic_router = semantic_router
        # The tool catalog (#1736): what this agent holds, advertises, binds and resolves.
        # It owns the agent-local instances, host binding (design §5.1) and `search_tools`;
        # everything it reads back from this agent is read live through the scope.
        self._tool_invoker = ToolInvoker(
            ToolScope(
                registry=lambda: self._tools,
                allowed_tools=lambda: self._config.allowed_tools,
                room_id=lambda: self._turn_room_id,
                capability_refusal=self._capability_refusal,
                may_call_peers=self._may_call_peers,
                live_session=self._live_session,
            ),
            local_tools=agent_local_tools,
            agent_bound_types=AGENT_BOUND_TOOL_TYPES,
            binder=tool_binder,
        )
        # Lambdas, not bound methods: tests patch these on the instance, and the harness
        # ladder swaps `_tools` after construction; the assembler must see both.
        self._prompt_assembler = PromptAssembler(
            PromptScope(
                ontology=lambda: self._ontology,
                skills=lambda: self._skills,
                config=lambda: self._config,
                tools=lambda: self._tools,
                memory=lambda: self._memory,
                workspace_root=lambda: self._resolve_workspace_root(),
                current_plan=lambda: self.current_plan,
                history=lambda: self._history,
                active_session=lambda: self._active_session,
                turn_counter=lambda: self._turn_counter,
                anchor_is_stale=lambda: self._anchor_is_stale(self._active_session),
                system_prompt_base=lambda: self._system_prompt_base(),
                effective_system_prompt=lambda: self.effective_system_prompt,
            )
        )
        self._tool_call_executor = ToolCallExecutor(
            ExecutionScope(
                agent_id=lambda: self.agent_id,
                context=lambda: self._context,
                config=lambda: self._config,
                tool_invoker=lambda: self._tool_invoker,
                hook_runner=lambda: self._hook_runner,
                approvals_answered=lambda: self._approvals_answered,
                bus=lambda: self._bus,
                publisher=lambda: self._publisher,
                capability_refusal=lambda tool: self._capability_refusal(tool),
                current_plan=lambda: self.current_plan,
                create_plan=lambda title, steps: self.create_plan(title=title, steps=steps),
                update_step_status=lambda index, completed, verification: self.update_step_status(
                    index=index, completed=completed, verification=verification
                ),
                live_session=lambda session_id: self._live_session(session_id),
                publish_plan_update=lambda plan: self._publish_plan_update(plan),
            )
        )
        self._turn_executor = TurnExecutor(
            TurnScope(
                agent_id=lambda: self.agent_id,
                llm=lambda: self._llm,
                config=lambda: self._config,
                context=lambda: self._context,
                tracer=lambda: self._tracer,
                budget=lambda: self._budget,
                tool_registry=lambda: self._tools,
                skills=lambda: self._skills,
                loaded_skills=lambda: self._loaded_skills,
                state=lambda: self._state,
                turn_lock=lambda: self._turn_lock,
                hook_runner=lambda: self._hook_runner,
                tool_invoker=lambda: self._tool_invoker,
                prompt_assembler=lambda: self._prompt_assembler,
                semantic_router=lambda: self._semantic_router,
                plan_generator=lambda: self._plan_generator,
                publisher=lambda: self._publisher,
                bus=lambda: self._bus,
                history=lambda: self._history,
                active_session=lambda: self._active_session,
                pending_durable_events=lambda: self._pending_durable_events,
                persona_name=lambda: self.persona_name,
                persona=lambda: self.persona,
                workspace_root=lambda: self.workspace_root,
                agent_delegate=lambda: self,
                turn_counter=lambda: self._turn_counter,
                set_turn_counter=lambda value: setattr(self, "_turn_counter", value),
                run_steps=lambda: self._run_steps,
                set_run_steps=lambda value: setattr(self, "_run_steps", value),
                turn_room_id=lambda: self._turn_room_id,
                set_turn_room_id=lambda value: setattr(self, "_turn_room_id", value),
                turn_caller_turn_id=lambda: self._turn_caller_turn_id,
                set_turn_caller_turn_id=lambda value: setattr(self, "_turn_caller_turn_id", value),
                turn_story_id=lambda: self._turn_story_id,
                set_turn_story_id=lambda value: setattr(self, "_turn_story_id", value),
                transition_to=lambda: self.transition_to,
                live_session=lambda: self._live_session,
                active_skill_dirs=lambda: self.active_skill_dirs,
                context_window=lambda: self._context_window,
                reply_reserve=lambda: self._reply_reserve,
                resolve_workspace_root=lambda: self._resolve_workspace_root,
                resolve_tool_isolation=lambda: self._resolve_tool_isolation,
                prepare_turn_layers=lambda: self._prepare_turn_layers,
                prepare_turn_messages=lambda: self._prepare_turn_messages,
                nudged_retry=lambda: self._nudged_retry,
                image_set_section=lambda: self._image_set_section,
                case_skill_section=lambda: self._case_skill_section,
                auto_compact_if_needed=lambda: self._auto_compact_if_needed,
                ingest_tool_message=lambda: self._ingest_tool_message,
                execute_single_tool=lambda: self._execute_single_tool,
                after_tool_step=lambda: self._after_tool_step,
                invoke_model=lambda: self._invoke_model,
                fit_step_to_window=lambda: self._fit_step_to_window,
                execute_tools=lambda: self._execute_tools,
                take_staged_persona=lambda: self._take_staged_persona,
            )
        )
        self._session_lifecycle = SessionLifecycle(
            SessionScope(
                agent_id=lambda: self.agent_id,
                config=lambda: self._config,
                context=lambda: self._context,
                set_context=lambda value: setattr(self, "_context", value),
                store=lambda: self._store,
                sessions=lambda: self._sessions,
                turn_lock=lambda: self._turn_lock,
                running=lambda: self._running,
                subscription=lambda: self._subscription,
                stranded_event_counts=lambda: self._stranded_event_counts,
                pending_durable_events=lambda: self._pending_durable_events,
                set_pending_durable_events=lambda value: setattr(
                    self, "_pending_durable_events", value
                ),
                session_compactors=lambda: self._session_compactors,
                run_steps=lambda: self._run_steps,
                set_run_steps=lambda value: setattr(self, "_run_steps", value),
                loaded_skills=lambda: self._loaded_skills,
                tool_invoker=lambda: self._tool_invoker,
                effective_system_prompt=lambda: self.effective_system_prompt,
                resolved_persona=lambda: self._resolved_persona,
                resolve_workspace_root=lambda: self._resolve_workspace_root,
                effective_session_id=lambda: self._effective_session_id,
                refuse_session_mutation_during_turn=lambda: (
                    self._refuse_session_mutation_during_turn
                ),
                seed_live_session=lambda: self._seed_live_session,
                live_session=lambda: self._live_session,
                get_session=lambda: self.get_session,
                subscription_topics=lambda: self._subscription_topics,
                write_pending_bodies=lambda: self._write_pending_bodies,
            )
        )
        self._compaction_driver = CompactionDriver(
            CompactionScope(
                agent_id=lambda: self.agent_id,
                config=lambda: self._config,
                context=lambda: self._context,
                tool_registry=lambda: self._tools,
                store=lambda: self._store,
                budget=lambda: self._budget,
                publisher=lambda: self._publisher,
                tracer=lambda: self._tracer,
                processing_errors=lambda: self._processing_errors,
                tool_invoker=lambda: self._tool_invoker,
                injected_compactor=lambda: self._injected_compactor,
                session_compactors=lambda: self._session_compactors,
                history=lambda: self._history,
                live_session=lambda: self._live_session,
                effective_session_id=lambda: self._effective_session_id,
                refuse_session_mutation_during_turn=lambda: (
                    self._refuse_session_mutation_during_turn
                ),
                write_pending_bodies=lambda: self._write_pending_bodies,
                resolve_workspace_root=lambda: self._resolve_workspace_root,
                context_window=lambda: self._context_window,
                explicit_threshold=lambda: self._explicit_threshold,
                observe_context_window=lambda: self._observe_context_window,
                prepare_turn_messages=lambda: self._prepare_turn_messages,
                session_compactor=lambda: self._session_compactor,
                compact_session=lambda: self._compact_session,
                should_compact_session=lambda: self._should_compact_session,
            )
        )
        self._plan_generator = plan_generator
        # One compactor per session when none is injected. See `_session_compactor`.
        self._session_compactors: dict[str, ContextCompactorProtocol] = {}
        self._state = AgentState.IDLE
        self._context = context or AgentContext(
            session_id=f"sess_{config.agent_id}",
            agent_id=config.agent_id,
            current_state=self._state,
        )
        # Persona resolution needs the workspace to reach `.uclone/personas/`, and the
        # context that carries it is assigned here. Resolving earlier -- as this did, from
        # `__init__` before `_context` existed, behind a `hasattr` guard that was therefore
        # always false -- silently searched only the shipped personas, so a workspace
        # persona could never contribute its tools. Order is the fix; the guard hid it.
        self._workspace_root_hint: Path | None = self._context.workspace_root
        # The tool restriction the *operator* asked for, kept apart from whatever a persona
        # contributes. `_config.allowed_tools` is the resolved value the tool scope is read
        # from -- 3 sites, by `git grep -n 'self\._config\.allowed_tools' 4e5a2844 --
        # src/uclone_x/agent/base.py`: one filters the tool list the model is offered
        # (`:2379`), one refuses a call inside a turn (`:3344`), one raises `PermissionError`
        # from a direct call (`:3699`). This is the input all three are resolved *from*, so
        # a later persona change recomputes rather than layering the new persona's list onto
        # the one the previous persona contributed (#1081).
        self._operator_allowed_tools: tuple[str, ...] = config.allowed_tools
        self._apply_persona_tool_scope()
        if tracer is not None:
            self._tracer = tracer
        elif self._context.trace_id:
            self._tracer = TelemetryTracer(trace_id=self._context.trace_id)
        else:
            self._tracer = TelemetryTracer()
        if self._context.trace_id is None:
            self._context = self._context.model_copy(update={"trace_id": self._tracer.trace_id})
        # Multi-session state (#183, requirement 1). One `_LiveSession` per session id;
        # the active one is named by `self._context.session_id`. Seeding goes through
        # `SessionState.seed`, which is the same single reset semantics `reset_session`
        # uses, so a fresh session and a reset session are constructed by one code path
        # rather than two that can drift.
        self._sessions: dict[str, _LiveSession] = {
            self._context.session_id: self._seed_live_session(self._context.session_id)
        }
        self._subscription: EventSubscriptionProtocol | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._publisher: PublisherHandleProtocol | None = None
        if self._bus is not None:
            self._publisher = self._bus.register_publisher(
                sender_id=self.agent_id,
                source=EventSource.AGENT,
            )
        if hook_runner is not None:
            self._hook_runner = hook_runner
            if hooks:
                self._hook_runner.register_hooks(hooks)
            elif self._config.hooks:
                self._hook_runner.register_hooks(self._config.hooks)
        else:
            initial_hooks: list[BaseHook] = []
            if self._config.hooks:
                initial_hooks.extend(self._config.hooks)
            if hooks:
                initial_hooks.extend(hooks)
            self._hook_runner = HookRunner(
                hooks=initial_hooks,
                bus=self._bus,
                publisher=self._publisher,
            )
        self._running = False
        self._turn_lock = asyncio.Lock()
        # Bounded so a long-lived agent absorbing a repeating failure cannot grow this
        # without limit; the newest failures are the diagnostic ones.
        self._processing_errors: deque[BaseException] = deque(maxlen=_MAX_RECORDED_ERRORS)
        # Events discarded by a session switch, keyed by the session they belonged to
        # (#225). A count rather than the events themselves: retaining them would grow
        # without bound, and re-delivering a `USER_INPUT` after a later switch back would
        # answer a question the user asked in a different context minutes earlier. What
        # P6 requires is that the discard be observable and attributable, not reversible.
        self._stranded_event_counts: dict[str, int] = {}

    @property
    def hook_runner(self) -> HookRunner:
        """Active HookRunner orchestrating lifecycle and tool execution hooks."""
        return self._hook_runner

    @property
    def agent_id(self) -> str:
        return self._config.agent_id

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def context(self) -> AgentContext:
        return self._context

    @property
    def workspace_root(self) -> Path | None:
        """Workspace root directory for the agent, if defined."""
        if self._context.workspace_root is not None:
            return self._context.workspace_root.resolve()
        if self._workspace_root_hint is not None:
            return self._workspace_root_hint.resolve()
        return None

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def persona(self) -> str | None:
        """Name of the governing persona, if configured."""
        return self._persona

    @property
    def persona_definition(self) -> PersonaDefinition | None:
        """The persona definition in force, or `None` (see `_resolved_persona`)."""
        return self._resolved_persona()

    @property
    def approvals_answered(self) -> bool:
        """Whether a person answers this agent's approval requests during a turn.

        `False` in the desktop app, which has no approval prompt in a conversation: a call
        that needs a person is refused at once there, rather than after the timeout.
        """
        return self._approvals_answered

    @property
    def a2a_transport(self) -> A2ATransportProtocol | None:
        """The transport `a2a_call` reaches peer personas through, or `None` (#1558).

        `None` is the agent that may not call another: an agent a peer call built is given
        none, which is what holds A2A calls to one level deep. A sub-agent is not given its
        parent's either (`SUBAGENT_EXCLUDED_HOST_FIELDS`).
        """
        return self._a2a_transport

    @persona.setter
    def persona(self, value: str | None) -> None:
        self._set_persona(value)

    @property
    def persona_name(self) -> str | None:
        """Alias for persona."""
        return self._persona

    @persona_name.setter
    def persona_name(self, value: str | None) -> None:
        # Delegates to the property it aliases, not to `_set_persona` alongside it: an
        # alias that reaches the same work by its own route is the shape that drifted.
        self.persona = value

    def _set_persona(self, value: str | None) -> None:
        """Adopt `value` as the governing persona, and recompute the tool scope it implies.

        Both setters route here so they cannot drift: they were two independent
        `self._persona = value` assignments, which is a shape that stays correct only
        while adopting a persona has no consequences beyond the name (#1081).

        A persona carries a prompt and a tool restriction, and a plain assignment moved
        only the prompt. `effective_system_prompt` re-resolves on every read, so it
        followed the assignment; the tool restriction was resolved by the `__init__` block
        `_apply_persona_tool_scope` replaced, and assigning to `_persona` does not re-run
        `__init__`. So an agent given a persona after construction adopted the persona's
        prompt while calling tools the persona withholds — the two halves disagreeing
        about whether the persona had been adopted at all.

        A session anchored before the adoption is the remaining consequence. It is not
        rewritten here — see `_prepare_turn_messages` for why the anchor is left saying
        what it actually said, and the persona axis re-resolved per turn instead.

        **Nothing about the anchor is decided here**, which is why this method is two
        statements and not three. Two earlier revisions of this fix recorded something at
        this call — first "an assignment happened", then "an assignment moved the resolved
        persona" — and both were measured wrong on the wire, at constructions the PR body
        lists: the first discarded a caller's hydrated anchor at C1 and C2, the second at
        C8 and C9. What the turn builder needs is which axis position composed the anchor
        it is looking at, and that is knowable where the anchor is written, so it is
        stamped there, on the session, as `anchor_provenance`.
        """
        self._persona = value
        self._apply_persona_tool_scope()

    def _resolved_persona(self) -> PersonaDefinition | None:
        """The persona definition in force, or `None` when the axis resolves to nothing.

        A name `get_persona` returns nothing for resolves to nothing, exactly as a
        cleared name does. Collapsing "no name" and "a name that resolves to no
        definition" into one `None` is what lets `_anchor_is_stale` compare two
        resolutions instead of two names — which is also why an anchor stays fresh across
        `a -> no_such_name`, and goes stale when `define_persona` gives a name a
        definition it did not have.
        """
        if not self._persona:
            return None
        return self.get_persona(self._persona)

    def _anchor_is_stale(self, session: _LiveSession) -> bool:
        """Whether `session`'s anchored system turn still says what the axis resolves to.

        The comparison is between two *resolutions*, not two names, so a name that
        resolves to nothing sits on the same side as no name at all.

        A caller-composed anchor is reported fresh whatever the axis says. The agent did
        not write that text and cannot know which axis position it came from, so there is
        no position to compare against — and replacing it would discard something a caller
        supplied deliberately, which is the mode #1081 records as measured and rejected
        for PR #937. An anchor restored from a record that does not say what composed it
        is reported fresh for the same reason and not the same one: there is no position
        to compare, so there is nothing this method can conclude. `hydrate_session` is
        where that absence is reported (#1152); silence here is the honest answer, and
        the *stale* answer would be the invention.
        """
        if isinstance(session.anchor_provenance, AnchorWriter):
            return False
        if session.anchor_provenance != self._resolved_persona():
            return True
        return self._anchor_framing_moved(session)

    def _anchor_framing_moved(self, session: _LiveSession) -> bool:
        """Whether `session`'s agent-composed anchor carries a seat framing no longer in force.

        A one-seat room seeds its clone without the multi-agent framing, and a second clone
        joining puts it back (§5.9.3); leaving takes it away again. The persona has not
        moved, so the persona comparison above calls such an anchor fresh, and the turn
        would go on sending the framing of a roster that is gone. The framing is read back
        from the anchor's text (`composed_seat_framing`). An anchor that is not a
        composition of this persona under any framing is left alone: this method cannot
        say what it was, and replacing it is the mode #1081 rejects.

        The anchor itself is not rewritten; the turn sends the recomposed identity, which
        costs one prompt-cache miss when the roster crosses between one clone and two.
        """
        if not session.messages or session.messages[0].role is not MessageRole.SYSTEM:
            return False

        def canonical(text: str) -> str:
            return adapt_system_prompt(text, None)

        persona = self._resolved_persona()
        if persona is not None:
            persona = persona.model_copy(update={"system_prompt": canonical(persona.system_prompt)})
        recorded = composed_seat_framing(
            canonical(session.messages[0].content or ""),
            config_prompt=canonical(self._config.system_prompt),
            persona=persona,
        )
        return recorded is not None and recorded != canonical(self._config.seat_framing)

    def set_seat_framing(self, framing: str) -> None:
        """Place this agent in a room whose roster changed: frame its seat with `framing`.

        The room decides the framing (`""` for a clone alone with its people, §5.9.3); the
        agent composes it into its identity as at construction. A session anchored under
        the other framing is caught by `_anchor_framing_moved` on its next turn.
        """
        if framing != self._config.seat_framing:
            self._config = self._config.model_copy(update={"seat_framing": framing})

    def _apply_persona_tool_scope(self) -> None:
        """Resolve `config.allowed_tools` from the operator's list and the persona's.

        **The rule, stated once because it is a behaviour choice and not an implementation
        detail:** an operator-supplied `allowed_tools` wins outright, and a persona's list
        applies only where the operator gave none. That is the rule `__init__` already
        applied at construction (`not config.allowed_tools`); routing both construction and
        the setter through here is what extends it to a persona adopted later, unchanged.

        The resolution runs from `_operator_allowed_tools` rather than from the current
        `config.allowed_tools`, which is what makes it a recomputation rather than an
        accumulation: persona A swapped for persona B yields B's list, and clearing the
        persona takes back the tools a persona contributed instead of leaving them in
        force under no persona at all.

        A persona that resolves but lists no tools contributes nothing, exactly as before:
        an empty list is the absence of a restriction here, not a restriction to nothing.

        A persona's list is read through `granted_tools`, which adds the base set every
        persona is given (`BASE_PERSONA_TOOLS`, #1402). An operator's list is taken as
        written: the operator is naming exactly what this agent may run.
        """
        self._config = self._persona_scoped_config(self._persona_store)

    def _persona_scoped_config(self, store: Mapping[str, PersonaDefinition]) -> AgentConfig:
        """The config `_apply_persona_tool_scope` would set were `store` the registrations.

        Reads the agent and changes nothing, so a registration can be worked out in full
        before any of it is applied (`_persona_definition_commit`).
        """
        resolved = self._operator_allowed_tools
        persona = self._persona_in(store, self._persona) if self._persona else None
        if not resolved and persona is not None and persona.granted_tools:
            resolved = persona.granted_tools
        if resolved == self._config.allowed_tools:
            return self._config
        return self._config.model_copy(update={"allowed_tools": resolved})

    @property
    def write_tools_enabled(self) -> bool:
        """Whether this agent may run a tool that writes files on the host (#1167).

        **The precedence, stated once:** the flags only ever remove tools. The agent's own
        `config.enable_write_tools` (the operator's, or the parent's for a sub-agent) and the
        persona in force are both consulted, and a `False` from either wins; neither can
        grant what the other withheld. They are independent of `allowed_tools`: a tool
        must pass both, so naming `file_write` in an operator's or a persona's
        `allowed_tools` does not lift `enable_write_tools: false`, and the flag being `True`
        does not add a tool the list leaves out.

        The persona is read as it resolves now, not as it was at construction, so an edit
        from the Settings editor reaches an agent that is already running -- the same rule
        `_apply_persona_tool_scope` gives `allowed_tools`.
        """
        persona = self._resolved_persona()
        operator_allows = self._config.enable_write_tools
        persona_allows = persona is None or persona.enable_write_tools
        return operator_allows and persona_allows

    @property
    def subagent_tools_enabled(self) -> bool:
        """Whether this agent may start another agent (#1167). Precedence as for writes."""
        persona = self._resolved_persona()
        operator_allows = self._config.enable_subagent_tools
        persona_allows = persona is None or persona.enable_subagent_tools
        return operator_allows and persona_allows

    def _capability_refusal(self, tool: object) -> str | None:
        """Why the persona flags withhold `tool` from this agent, or `None` if they do not.

        One answer read by every site that decides whether a tool is available -- the
        advertisement, the refusal inside a turn and `execute_tool_call` -- so what the
        model is offered and what it is allowed to run cannot drift apart. The words name
        the flag that refused and where to change it, because "not allowed" alone tells a
        reader nothing they can act on.
        """
        name = getattr(tool, "name", "?")
        if tool_writes_files(tool) and not self.write_tools_enabled:
            return (
                f"Tool '{name}' can write files, and agent '{self.agent_id}' has "
                "enable_write_tools off (turn on 'Allow file-writing tools' in the "
                "persona's settings to allow it)"
            )
        if tool_spawns_subagents(tool) and not self.subagent_tools_enabled:
            return (
                f"Tool '{name}' starts a sub-agent, and agent '{self.agent_id}' has "
                "enable_subagent_tools off (turn on 'Allow sub-agents' in the persona's "
                "settings to allow it)"
            )
        return None

    def _system_prompt_base(self) -> str:
        """The prompt the persona axis resolves to, before any model-family framing.

        Answers "which prompt does the persona in force select?", and is read by both
        `effective_system_prompt` and the turn builder, so the property and the system
        message actually sent resolve the persona from one expression (#1081).
        """
        return compose_identity_prompt(
            config_prompt=self._config.system_prompt,
            persona=self._resolved_persona(),
            seat_framing=self._config.seat_framing,
        )

    @property
    def effective_system_prompt(self) -> str:
        """Resolve the effective system prompt for the active persona and model family.

        Two steps, in order. The persona (when one is active) selects the base prompt,
        exactly as before. `adapt_system_prompt` then re-frames that prompt's
        steerability section for the configured model family (#871) — a no-op for every
        family without its own framing, and a no-op for any prompt an operator wrote
        that does not embed the default components.
        """
        base = self._system_prompt_base()
        return adapt_system_prompt(base, self._config.llm_config.model_name)

    async def step(self, input_data: str | AgentEvent) -> TurnResult:
        """[Deprecated alias for execute_turn] Execute a single reasoning turn."""
        return await self.execute_turn(input_data)

    async def run_turn(self, input_data: str | AgentEvent) -> TurnResult:
        """Execute a single reasoning turn (alias for execute_turn)."""
        return await self.execute_turn(input_data)

    @property
    def llm(self) -> LLMProviderProtocol | None:
        """Active LLM provider connector instance."""
        return self._llm

    async def invoke_auxiliary_model(self, request: LLMRequest) -> ModelResponse:
        """One model call outside a turn, on this clone's own connector and budget (#1404).

        For work done on the clone's behalf after a turn, such as learning from it: the
        call is budget-checked before and charged after, exactly as a turn's own calls are
        (P5). A request that names no model is sent with the clone's configured model, and
        with its configured context window when it names none.

        Raises:
            RuntimeError: The agent has no model connector.
        """
        llm = self._llm
        if llm is None:
            raise RuntimeError(f"Agent {self._config.agent_id} has no model connector")
        update: dict[str, Any] = {}
        if request.model is None and self._config.llm_config.model_name:
            update["model"] = self._config.llm_config.model_name
        limit = self._config.llm_config.context_limit
        if request.context_window is None and (limit or 0) > 0:
            update["context_window"] = limit
        return await self._invoke_model(
            llm, request.model_copy(update=update) if update else request
        )

    def set_read_roots(self, read_roots: tuple[Path, ...]) -> None:
        """Replace the folders outside the workspace this agent's read-only tools may read.

        Takes effect on the next tool call and the next turn's `[Workspace]` section, so a
        Settings change reaches a live conversation without a restart.
        """
        if read_roots != self._config.read_roots:
            self._config = self._config.model_copy(update={"read_roots": read_roots})

    def hot_reload_llm(
        self,
        llm: LLMProviderProtocol | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        fast_model: str | None = None,
    ) -> None:
        """Hot-reload active LLM connector and optional model or system prompt configuration without restart (Issue #350, #871).

        `model_name` is the deep model a turn runs on and `fast_model` the one auxiliary
        calls use; `None` leaves that slot as it is.
        """
        if llm is not None:
            self._llm = llm
        updates: dict[str, Any] = {}
        model_updates: dict[str, Any] = {}
        if model_name is not None:
            model_updates["model_name"] = model_name
        if fast_model is not None:
            model_updates["fast_model"] = fast_model
        if model_updates:
            updates["llm_config"] = self._config.llm_config.model_copy(update=model_updates)
        if system_prompt is not None:
            updates["system_prompt"] = system_prompt
        if updates:
            self._config = self._config.model_copy(update=updates)

    @property
    def ontology(self) -> OntologyEngineProtocol | None:
        return self._ontology

    @property
    def skills(self) -> SkillRegistryProtocol | None:
        """Active skill registry if configured (P9)."""
        return self._skills

    def _record_loaded_skill(self, skill_name: str) -> None:
        """Record an approved skill loaded into session context."""
        self._loaded_skills.add(skill_name)

    @property
    def loaded_skills(self) -> frozenset[str]:
        """Names of skills loaded into session context (P9)."""
        return frozenset(self._loaded_skills)

    def active_skill_dirs(self) -> tuple[Path, ...]:
        """The package folders of this agent's active skills, for `ToolContext.skill_dirs`.

        Read from the registry each time, so a skill approved or reloaded between steps
        counts from the next one. A skill with no folder on disk is left out.
        """
        if self._skills is None:
            return ()
        dirs: list[Path] = []
        for skill in self._skills.list_skills():
            if skill.manifest.status != SkillStatus.ACTIVE:
                continue
            directory: object = getattr(skill, "directory", None)
            if isinstance(directory, Path):
                dirs.append(directory)
        return tuple(dirs)

    async def reload_skills(self) -> tuple[SkillProtocol, ...]:
        """Hot-reload approved skills from disk into the agent's active registry (P9)."""
        if self._skills is None:
            return ()
        return await self._skills.reload_approved()

    @property
    def tracer(self) -> TracerProtocol:
        return self._tracer

    @property
    def store(self) -> SessionStoreProtocol | None:
        """The Core session store this agent persists through, if one is wired."""
        return self._store

    @property
    def tools(self) -> ToolRegistryProtocol | None:
        """Active tool registry if configured."""
        return self._tools

    @property
    def memory(self) -> CrossSessionMemory | None:
        """Active cross-session memory store."""
        return self._memory

    @property
    def current_plan(self) -> PlanState | None:
        """The active execution plan for this agent, if any, via session state."""
        return self._live_session(self._context.session_id).plan

    def create_plan(
        self,
        title: str,
        steps: Sequence[str | PlanStep | Mapping[str, Any]],
        plan_id: str | None = None,
    ) -> PlanState:
        """Create and initialize a new interactive chat plan (Antigravity adoption)."""
        norm_steps: list[PlanStep] = []
        for idx, s in enumerate(steps, start=1):
            if isinstance(s, PlanStep):
                norm_steps.append(s)
            elif isinstance(s, Mapping):
                norm_steps.append(
                    PlanStep(
                        index=int(s.get("index", idx)),
                        description=str(s.get("description", "")),
                        completed=bool(s.get("completed", False)),
                        verification=(
                            str(s["verification"]) if s.get("verification") is not None else None
                        ),
                    )
                )
            else:
                norm_steps.append(PlanStep(index=idx, description=str(s)))

        try:
            loop = asyncio.get_running_loop()
            ts = int(loop.time() * 1000)
        except RuntimeError:
            ts = 0

        pid = plan_id or f"plan_{self.agent_id}_{ts}"
        plan = PlanState(
            plan_id=pid,
            title=title,
            steps=tuple(norm_steps),
            status="proposed",
        )
        self._live_session(self._context.session_id).plan = plan
        self._publish_plan_update(plan)
        return plan

    def update_step_status(
        self,
        index: int,
        completed: bool = True,
        verification: str | None = None,
    ) -> PlanState:
        """Update the completion status and verification criteria of a step."""
        current_plan = self.current_plan
        if current_plan is None:
            raise RuntimeError("No active plan exists to update step status")

        new_steps: list[PlanStep] = []
        found = False
        for s in current_plan.steps:
            if s.index == index:
                found = True
                new_steps.append(
                    PlanStep(
                        index=s.index,
                        description=s.description,
                        completed=completed,
                        verification=verification if verification is not None else s.verification,
                    )
                )
            else:
                new_steps.append(s)

        if not found:
            raise IndexError(
                f"Plan step with index {index} not found in plan '{current_plan.plan_id}'"
            )

        all_completed = all(step.completed for step in new_steps)
        status: Literal["proposed", "in_progress", "completed", "rejected"] = (
            "completed" if all_completed else "in_progress"
        )

        plan = PlanState(
            plan_id=current_plan.plan_id,
            title=current_plan.title,
            steps=tuple(new_steps),
            status=status,
        )
        self._live_session(self._context.session_id).plan = plan
        self._publish_plan_update(plan)
        return plan

    def clear_plan(self) -> None:
        """Clear and remove the currently active plan."""
        self._live_session(self._context.session_id).plan = None

    def _publish_plan_update(self, plan: PlanState) -> None:
        """Publish a PLAN_STATUS_UPDATE event onto the EventBus if wired."""
        if self._bus is None:
            return
        evt = AgentEvent(
            type=EventType.PLAN_STATUS_UPDATE,
            sender_id=self.agent_id,
            topic=f"session.{self._context.session_id}",
            payload={
                "plan_id": plan.plan_id,
                "title": plan.title,
                "status": plan.status,
                "steps": [
                    {
                        "index": s.index,
                        "description": s.description,
                        "completed": s.completed,
                        "verification": s.verification,
                    }
                    for s in plan.steps
                ],
            },
            trace_id=self._tracer.trace_id,
        )
        try:
            loop = asyncio.get_running_loop()
            if self._publisher is not None:
                loop.create_task(self._publisher.publish(evt))
            else:
                loop.create_task(self._bus.publish(evt))
        except RuntimeError:
            pass

    # ----------------------------------------------------------------------------------
    # Active-session views. `_history` and `_turn_counter` are properties rather than
    # fields so that the active session's messages and turn count have exactly one
    # storage location — the `_LiveSession` in `_sessions` — and cannot drift from it.
    # They stay private-by-name because `cli/commands/run.py` reaches for `_history`
    # through a `reportPrivateUsage` pragma today; those pragmas go away with the CLI
    # seam, and the public API below is what replaces them.
    # ----------------------------------------------------------------------------------

    @property
    def _active_session(self) -> _LiveSession:
        """The live session named by `context.session_id`, created on demand."""
        return self._live_session(self._context.session_id)

    @property
    def _history(self) -> list[ChatMessage]:
        return self._active_session.messages

    @_history.setter
    def _history(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> None:
        # What is being replaced is logged first: it leaves the history, not the log. What
        # replaces it is logged at once, so the count of what the history holds is current
        # before anything else enters (#1443).
        self._active_session.log_history()
        self._active_session.messages = list(messages)
        self._active_session.log_history()  # and what replaced it
        self._active_session.declare_new_epoch("history_replaced")
        self._active_session.updated_at = _now_iso()

    @property
    def _turn_counter(self) -> int:
        return self._active_session.turn_counter

    @_turn_counter.setter
    def _turn_counter(self, value: int) -> None:
        self._active_session.turn_counter = value

    @property
    def run_steps(self) -> int:
        """Agent steps taken in the current — or, between requests, the most recent — run.

        This, not `turn_counter`, is the quantity `AgentConfig.max_steps` bounds: one model
        invocation and its tool round, taken without returning to whoever asked. A surface
        that reports a budget must report the quantity the ceiling is measured against, or
        it shows a person a bar filling toward a limit that will never be reached.
        """
        return self._run_steps

    @property
    def run_turns(self) -> int:
        """[Deprecated alias for run_steps] Agent steps taken in the current run."""
        return self.run_steps

    @property
    def steps_remaining(self) -> int:
        """Agent steps remaining in the current run before reaching max_steps ceiling."""
        return max(0, self._config.max_steps - self.run_steps)

    @property
    def turns_remaining(self) -> int:
        """[Deprecated alias for steps_remaining] Steps remaining in the current run."""
        return self.steps_remaining

    def consume_steps(self, count: int) -> None:
        """Consume steps from the current run budget (e.g. charged by child delegations per P4)."""
        if count > 0:
            self._run_steps += count

    @property
    def turn_counter(self) -> int:
        """The active session's turn counter."""
        return self._turn_counter

    # The session lifecycle (#1736): `SessionLifecycle` in `agent/session_lifecycle.py`
    # holds the moved bodies. These stay as delegators because other modules and tests
    # call them, and the moved code calls them back through this agent.

    def _effective_session_id(self, session_id: str | None) -> str:
        """Resolve an optional session argument to a concrete id (`None` is the active one)."""
        return self._session_lifecycle.effective_session_id(session_id)

    def _refuse_session_mutation_during_turn(self, session_id: str, operation: str) -> None:
        """Refuse a reset or switch that would corrupt a turn in flight **on this agent**."""
        self._session_lifecycle.refuse_session_mutation_during_turn(session_id, operation)

    def _seed_live_session(self, session_id: str) -> _LiveSession:
        """A freshly seeded live session, stamped with the axis position it was seeded at."""
        return self._session_lifecycle.seed_live_session(session_id)

    def _live_session(self, session_id: str) -> _LiveSession:
        """Return the live session for `session_id`, seeding it if it is new."""
        return self._session_lifecycle.live_session(session_id)

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return the immutable conversation history of the **active** session."""
        return tuple(self._history)

    @property
    def session_id(self) -> str:
        """The active session's identifier."""
        return self._context.session_id

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Every session this agent is currently hosting, sorted."""
        return tuple(sorted(self._sessions))

    def get_session(self, session_id: str | None = None) -> SessionState:
        """Snapshot one hosted session as frozen `SessionState`, without seeding it."""
        return self._session_lifecycle.get_session(session_id)

    def _subscription_topics(self, session_id: str) -> set[str]:
        """The topic set this agent listens on while active in `session_id`."""
        return self._session_lifecycle.subscription_topics(session_id)

    @property
    def stranded_event_counts(self) -> Mapping[str, int]:
        """Events discarded by a session switch, keyed by the session they belonged to.

        A switch on a *running* agent repoints the bus subscription. Events already
        queued for the session being left cannot be answered after the switch — the turn
        would run against the new conversation — so they are discarded. P6 forbids a
        failure whose only trace is a log line or a sentence in a docstring, so the
        discard is counted here, attributed to the session that lost them, and mirrored
        on the subscription's own `drop_reasons["retarget_stranded"]`.

        An empty mapping means no switch has discarded anything, which is the ordinary
        case: events on `agent.{id}` and `broadcast` survive a switch, so only traffic
        addressed to the departing session is ever at risk.
        """
        return dict(self._stranded_event_counts)

    def switch_session(self, session_id: str) -> SessionState:
        """Make `session_id` the active session, seeding it if new (see `SessionLifecycle`)."""
        return self._session_lifecycle.switch_session(session_id)

    def load_history(
        self,
        messages: list[ChatMessage] | tuple[ChatMessage, ...],
        turn_counter: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """Hydrate one session's messages and turn counter from a persisted session."""
        self._session_lifecycle.load_history(messages, turn_counter, session_id)

    @property
    def pending_durable_events(self) -> tuple[Mapping[str, Any], ...]:
        """What the last turns recorded, not yet drained by `persist_session`.

        Read-only and a copy: the queue is the agent's, and a caller that could mutate it
        could remove the record of its own turn. Public because "why did this turn stop"
        is a question callers ask -- the eval harness asks it of every problem -- and the
        answer was reachable only through a private attribute.
        """
        return tuple(dict(event) for event in self._pending_durable_events)

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        """The session as it stands before a turn, for `roll_back_turn` to return to (#1423)."""
        return self._session_lifecycle.checkpoint_turn(session_id)

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        """Return a session's conversation to `checkpoint`; the number of messages dropped."""
        return self._session_lifecycle.roll_back_turn(checkpoint, reason=reason)

    def reset_session(self, session_id: str | None = None) -> SessionState:
        """Purge a session back to its seeded state and zero its turn counter."""
        return self._session_lifecycle.reset_session(session_id)

    def persist_session(
        self,
        session_id: str | None = None,
        *,
        pending_events: Sequence[Any] | None = None,
    ) -> SessionState:
        """Write one session to the Core store."""
        return self._session_lifecycle.persist_session(session_id, pending_events=pending_events)

    def _write_pending_bodies(self, sid: str) -> None:
        """Write the bodies `sid`'s snapshots name and the store does not hold yet."""
        self._session_lifecycle.write_pending_bodies(sid)

    def hydrate_session(self, session_id: str | None = None) -> SessionState | None:
        """Load one session from the Core store, replacing what is held in memory."""
        return self._session_lifecycle.hydrate_session(session_id)

    def delete_session(self, session_id: str | None = None) -> bool:
        """Delete a session's record and clean up associated tool artifacts (P3)."""
        return self._session_lifecycle.delete_session(session_id)

    # Context compaction (#183 requirement 3, P5, #1736): `CompactionDriver` in
    # `agent/compaction_driver.py` holds the moved bodies. These stay as delegators
    # because the turn loop and tests call them and tests patch them on the instance.

    def _session_compactor(self, session_id: str) -> ContextCompactorProtocol:
        """The compactor for one session: the injected one, else one per session id."""
        return self._compaction_driver.session_compactor(session_id)

    async def compact_session(
        self,
        session_id: str | None = None,
        reason: str = "manual_on_demand",
    ) -> CompactionResult:
        """Compact one session's context and publish a `CONTEXT_COMPACTED` notice (P5)."""
        return await self._compaction_driver.compact_session(session_id, reason)

    async def _compact_session(
        self,
        sid: str,
        reason: str,
        *,
        reader_offered: bool | None = None,
    ) -> CompactionResult:
        """Compact one session unconditionally, without the in-flight-turn guard."""
        return await self._compaction_driver.compact_session_unguarded(
            sid, reason, reader_offered=reader_offered
        )

    def _should_compact_session(
        self,
        session_id: str,
        messages: Sequence[ChatMessage],
        *,
        request: LLMRequest | None = None,
    ) -> bool:
        """Determine if a session needs auto-compaction based on configured threshold (P5)."""
        return self._compaction_driver.should_compact_session(session_id, messages, request=request)

    async def _auto_compact_if_needed(
        self,
        tools: Sequence[ToolDefinition] = (),
        extra_sections: Sequence[str] = (),
        *,
        reason: str = "auto_threshold",
    ) -> CompactionResult | None:
        """Compact the active session when the request about to be sent reaches the threshold."""
        return await self._compaction_driver.auto_compact_if_needed(
            tools, extra_sections, reason=reason
        )

    def _nudged_retry(
        self,
        req: LLMRequest,
        nudge: str,
        turn_extra_sections: Sequence[str],
    ) -> tuple[LLMRequest, RequestLayers]:
        """Build the request that re-asks after an evidence or grounding nudge (#1420).

        **The retry is built from history, not from the request that produced the answer.**
        That request predates the answer: the answer went to `_history`, never into
        `req.messages`. Appending the nudge to it sent a critique of an answer the model
        could not see, and ended the request in two consecutive `USER` messages (the
        span, then the nudge), which chat templates that require alternating roles refuse.
        Rebuilt the way a tool step is, the request ends
        `... ASSISTANT(rejected answer) · USER(turn context + nudge)`: the nudge is turn
        context (the request-layering design, §5.5), and `turn_context_block`
        places it so no two `USER` messages are adjacent.

        **What happens to the rejected answer is the loop's, not this method's:** it stays
        in history until the retry writes a message of its own, and is replaced by that
        message then (`superseded`, in `execute_turn`). The nudge never enters history
        (review of #702), so a rejected answer kept beside the retry's message would be two
        `ASSISTANT` turns in a row, in the record and in every later request, with the
        reason for the second one missing. A retry that writes nothing -- an empty reply,
        or a step that raises -- replaces nothing, so the rejected answer is still the
        turn's record and history never ends on the user's message. The rejected answer is
        also stored as its `ASSISTANT_MESSAGE` durable event, next to the nudge's own
        event.
        """
        layers = self._prepare_turn_layers(extra_sections=(*turn_extra_sections, nudge))
        retry_messages = assemble_request_messages(layers)
        return req.model_copy(update={"messages": tuple(retry_messages)}), layers

    def _prepare_turn_messages(self, extra_sections: Sequence[str] = ()) -> list[ChatMessage]:
        """The messages of the next request: `_prepare_turn_layers`, assembled."""
        return assemble_request_messages(self._prepare_turn_layers(extra_sections))

    def _prepare_turn_layers(self, extra_sections: Sequence[str] = ()) -> RequestLayers:
        """The next request's layers: `PromptAssembler.prepare_turn_layers` (#1736).

        Kept on the agent so `_nudged_retry`, the step loop and the window fit all build a
        request through one seam a test can patch.
        """
        return self._prompt_assembler.prepare_turn_layers(extra_sections)

    def _get_workspace_prompt_section(self) -> str | None:
        """The `[Workspace]` section: `PromptAssembler.get_workspace_prompt_section`."""
        return self._prompt_assembler.get_workspace_prompt_section()

    def transition_to(self, new_state: AgentState) -> None:
        """Validate and apply a reactive lifecycle state transition."""
        allowed = VALID_TRANSITIONS.get(self._state, frozenset())
        if new_state not in allowed and new_state != self._state:
            raise InvalidStateTransitionError(
                f"Illegal state transition from {self._state.value} to {new_state.value}"
            )
        self._state = new_state
        self._context = self._context.model_copy(update={"current_state": new_state})

    async def start(self) -> None:
        """Start the agent event listener on the bus."""
        if self._running:
            return
        self._running = True
        self.transition_to(AgentState.IDLE)
        if self._bus is not None:
            if self._publisher is None:
                self._publisher = self._bus.register_publisher(
                    sender_id=self.agent_id,
                    source=EventSource.AGENT,
                )
            self._subscription = self._bus.subscribe(
                topics=self._subscription_topics(self._context.session_id),
                recipient_id=self.agent_id,
                session_id=self._context.session_id,
            )
            self._loop_task = asyncio.create_task(self._event_loop())

    async def stop(self) -> None:
        """Stop the agent and clean up subscriptions."""
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None
        self.transition_to(AgentState.TERMINATED)

    @property
    def processing_errors(self) -> tuple[BaseException, ...]:
        """Failures absorbed by the agent, most recent last (bounded, newest kept).

        P6 forbids a failure that is visible only in telemetry. The event loop and
        post-commit event publishing (such as compaction notice delivery) must not crash
        atomic state or drop error signals silently. Exceptions absorbed in these paths
        are recorded here as well as logged, giving callers and diagnostics/health APIs
        truthful failure visibility.
        """
        return tuple(self._processing_errors)

    async def _event_loop(self) -> None:
        """Background coroutine consuming events from subscription.

        One event the agent cannot handle must not stop it consuming the next: P1 makes
        this a reactive state machine, and a coroutine that dies inside `create_task`
        leaves `_running` True, the state reporting `IDLE`, every later event silently
        undelivered, and the traceback surfacing only as a GC-time "Task exception was
        never retrieved". So `process_event` is guarded per event.

        The guard is not a silent fallback: nothing is substituted for the failed event,
        it is dropped rather than answered, the traceback is logged, and the failure is
        recorded on `processing_errors`. The agent is driven through `ERROR` so the
        transition is observable on `context.current_state`, then back to `IDLE` —
        staying in `ERROR` would make the *next* `IDLE`-only transition illegal and turn
        one bad event into a permanently dead agent, which is the failure this guard
        exists to prevent.
        """
        if self._subscription is None:
            return
        try:
            while self._running:
                event = await self._subscription.get()
                try:
                    await self.process_event(event)
                except Exception as exc:
                    self._processing_errors.append(exc)
                    logger.exception(
                        "Agent %s could not process event %s (type=%s); dropping it and "
                        "continuing to consume",
                        self.agent_id,
                        event.event_id,
                        event.type.value,
                    )
                    if self._state is not AgentState.TERMINATED:
                        self.transition_to(AgentState.ERROR)
                        self.transition_to(AgentState.IDLE)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    async def process_event(self, event: AgentEvent) -> bool:
        """Process an incoming event reactively."""
        if event.recipient_id and event.recipient_id != self.agent_id:
            return False

        if event.type == EventType.USER_INPUT:
            res = await self.execute_turn(event)
            if self._bus is not None and event.sender_id:
                # P6: attribution is asserted producer-side, before the result leaves
                # this agent, and travels in the typed envelope field rather than as a
                # payload key by convention (issue #51). `require_provenance` raises
                # here if the turn produced a result it cannot attribute, so an
                # unattributable reply is never published at all.
                #
                # This guard is reachable on a real path, and must stay that way: it
                # only bites because `execute_turn` no longer substitutes a synthetic
                # `agent.core/BaseAgent` provenance for a connector that stated none.
                # Re-introducing that fallback would not "fix" anything here — it would
                # make this line dead code again while leaving the comment above it
                # reading as though something were enforced. `_event_loop` catches what
                # this raises, so the loop drops the event instead of dying.
                reply_payload: dict[str, Any] = {
                    "turn_index": res.turn_index,
                    "content": res.content,
                    "is_completed": str(res.is_completed),
                    "correlation_id": event.event_id,
                }
                if getattr(res, "router_tier", None):
                    reply_payload["router_tier"] = res.router_tier
                if res.error is not None:
                    # Without this the failed turn reaches a subscriber as
                    # `content=""` and `is_completed="False"` with no stated cause,
                    # which is a failure visible only in the publisher's own logs.
                    reply_payload["error"] = res.error
                if res.tool_executions:
                    reply_payload["tool_executions"] = [
                        {
                            "tool_call_id": te.tool_call_id,
                            "tool_name": te.tool_name,
                            "arguments": cast(dict[str, Any], unwrap_immutable(te.arguments)),
                            "output": unwrap_immutable(te.output),
                            "status": te.status,
                            "error": te.error,
                            "duration_ms": te.duration_ms,
                        }
                        for te in res.tool_executions
                    ]
                reply_event = event.create_response(
                    type=EventType.AGENT_REPLY,
                    recipient_id=event.sender_id,
                    topic=f"session.{self._context.session_id}",
                    payload=reply_payload,
                    provenance=require_provenance(res.provenance, "TurnResult"),
                )
                if self._publisher is not None:
                    await self._publisher.publish(reply_event)
                else:
                    await self._bus.publish(reply_event)
            return True
        elif event.type == EventType.TOOL_RESULT:
            return True
        elif event.type == EventType.INTERRUPT:
            self.transition_to(AgentState.IDLE)
            return True
        return False

    async def _invoke_model(
        self,
        llm: LLMProviderProtocol,
        req: LLMRequest,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        progress: StreamProgress | None = None,
    ) -> ModelResponse:
        """One model invocation, budget-checked before and charged after; see `TurnExecutor`."""
        return await self._turn_executor.invoke_model(
            llm, req, stream_callback=stream_callback, progress=progress
        )

    async def execute_turn(
        self,
        input_data: str | AgentEvent,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        caller_turn_id: str | None = None,
        room_id: str | None = None,
        story_id: str | None = None,
        person_names: tuple[str, ...] = (),
    ) -> TurnResult:
        """Execute a single reasoning turn with serialized execution lock (P4, Issue #60).

        `person_names` are the names the room's person goes by; each tool call this turn
        receives them on its `ToolContext`, so `record_memory_fact` files a fact under
        one of them under `user` (#1857).

        `room_id` and `story_id` are the conversation (room) this turn runs in and the
        story it has open (#1555). A room passes its id and its `story_id` for every seat,
        and each tool call this turn receives them on its `ToolContext`. The agent only
        carries `story_id`; what moves it between steps is a lifecycle hook (#1732) --
        `uclone_x.story.StoryLifecycleHook`, which every head composes in, moves it after a
        successful call of a tool declaring `opens_story`. The story a turn leaves open is
        `TurnResult.story_id`, which is what a room keeps (#1775).

        Raises `LLMConnectorNotConfiguredError` when no connector is wired (#136a). The
        method used to answer that case with `f"Ack: {content_input}"`, reported as
        `is_completed=True` with `provenance` naming `agent.core/BaseAgent` as both
        `requested` and `served_by`, so `degraded` computed `False`. A caller could not
        distinguish that from a model's answer, which is precisely the "hardcoded
        default ... returned as if it were a success" P6 forbids unconditionally; its
        Classification Procedure question 1 — "does the caller receive a value that no
        real execution of the requested operation produced?" — reaches *forbidden* and
        stops, so no `Provenance` value could have rescued it. The repair is therefore
        not a better attribution but the absence of a result: P6's "unrepairable
        failures propagate".
        """
        result = await self._turn_executor.execute_turn(
            input_data,
            stream_callback=stream_callback,
            caller_turn_id=caller_turn_id,
            room_id=room_id,
            story_id=story_id,
            person_names=person_names,
        )
        # Read with no `await` between the turn's end and here, so no next turn has yet
        # reset it. Stamped once rather than at each of the executor's six returns.
        return result.model_copy(update={"story_id": self._turn_story_id})

    async def _image_set_section(
        self,
        message: str,
        tool_defs: Sequence[ToolDefinition],
        *,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str | None:
        """The planned prompts for an image set `message` asks for, as a turn section.

        `None` -- and the turn runs as it would have -- unless the persona is one
        `IMAGE_SET_PERSONAS` names, `generate_image` is offered this turn, and the message
        asks for two or more varied images. A planning failure of any kind is logged and
        also answers `None`: the plan improves a turn and must never be what breaks one.
        No failure text reaches the person.
        """
        llm = self._llm
        if llm is None or self._persona not in IMAGE_SET_PERSONAS:
            return None
        if not any(tool.name == "generate_image" for tool in tool_defs):
            return None
        count = detect_image_set(message)
        if count is None:
            return None

        async def generate(request: LLMRequest) -> ModelResponse:
            return await self._invoke_model(llm, request)

        limit = self._config.llm_config.context_limit
        try:
            if emit is not None:
                await emit("status", {"status": "thinking", "detail": "Planning the image set..."})
            prompts = await plan_image_set(
                generate,
                message,
                count,
                model=self._config.llm_config.model_name or None,
                context_window=limit if (limit or 0) > 0 else None,
                earlier=self._image_set_earlier(message),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Image-set planning skipped for agent %s: %s: %s",
                self.agent_id,
                type(exc).__name__,
                exc,
            )
            return None
        logger.info("Image-set plan for agent %s: %d prompts", self.agent_id, len(prompts))
        return image_set_note(prompts)

    async def _case_skill_section(
        self,
        message: str,
        tool_defs: Sequence[ToolDefinition],
        *,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str | None:
        """The case skills that fit `message`, as a turn section after the user's message.

        `None` unless the persona is one `CASE_SKILL_PERSONAS` names and `generate_image`
        is offered this turn. A follow-up (an image already drawn in this conversation) is
        read from history; a first turn is routed by grounded extraction, one structured
        call. An extraction failure of any kind is logged and answers `None`: routing
        improves a turn and must never be what breaks one. No failure text reaches the
        person.
        """
        llm = self._llm
        if llm is None or self._persona not in CASE_SKILL_PERSONAS:
            return None
        if not any(tool.name == "generate_image" for tool in tool_defs):
            return None
        if is_follow_up(self._history):
            names = route_follow_up(message)
        else:

            async def generate(request: LLMRequest) -> ModelResponse:
                return await self._invoke_model(llm, request)

            limit = self._config.llm_config.context_limit
            try:
                if emit is not None:
                    await emit("status", {"status": "thinking", "detail": "Reading the request..."})
                facts = await extract_request_facts(
                    generate,
                    message,
                    model=self._config.llm_config.model_name or None,
                    context_window=limit if (limit or 0) > 0 else None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Case-skill routing skipped for agent %s: %s: %s",
                    self.agent_id,
                    type(exc).__name__,
                    exc,
                )
                return None  # an unread request is drawn unrouted
            names = route_first_turn(grounded_facts(facts, message))
        logger.info("Case skills for agent %s: %s", self.agent_id, ", ".join(names))
        return case_skill_section(names)

    def _image_set_earlier(self, message: str, *, turns: int = 8, chars: int = 3000) -> str:
        """The recent conversation as plain lines, for the image-set planner.

        The last `turns` user, assistant and tool messages before `message`, each cut to
        400 characters, the whole kept to its last `chars`. Enough for a character settled
        a few turns back or a character sheet just read; the planner is a side call and
        must stay small.
        """
        picked: list[str] = []
        for entry in reversed(self._history):
            if entry.role is MessageRole.SYSTEM or not entry.content:
                continue
            if not picked and entry.role is MessageRole.USER and entry.content == message:
                continue
            picked.append(f"{entry.role.value}: {entry.content.strip()[:400]}")
            if len(picked) >= turns:
                break
        return "\n".join(reversed(picked))[-chars:]

    def _resolve_workspace_root(self) -> Path | None:
        """Resolve the effective workspace root boundary for tool execution."""
        if self._context.workspace_root is not None:
            return self._context.workspace_root.resolve()
        if self._config.workspace_dir is not None:
            return Path(self._config.workspace_dir).resolve()

        # If the host supplies a workspace, use it
        if self._host.workspace is not None:
            return self._host.workspace.root.resolve()

        return None

    def _resolve_tool_isolation(self) -> IsolationPolicy:
        """Resolve the effective isolation policy, clamped to at least the host floor."""
        requested_level = self._config.isolation.level

        floor = self._host.isolation_floor

        from uclone_x.sandbox.models import AVAILABLE_ISOLATION_LEVELS

        eff_level = effective_isolation_level(
            requested=requested_level,
            floor=floor,
            available_levels=self._host.available_isolation
            if self._host
            else AVAILABLE_ISOLATION_LEVELS,
        )
        if eff_level == requested_level:
            return self._config.isolation

        if eff_level == IsolationLevel.WORKSPACE:
            return WorkspaceIsolation()
        elif eff_level == IsolationLevel.WASM:
            from uclone_x.sandbox.models import WasmIsolation

            return WasmIsolation()
        elif eff_level == IsolationLevel.CONTAINER:
            raise NotImplementedError("Cannot auto-upgrade to ContainerIsolation without an image")

        return self._config.isolation

    async def _execute_single_tool(
        self,
        tc: ToolCallRequest,
        tool_ctx: ToolContext,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        advertised: frozenset[str] | None = None,
    ) -> tuple[ChatMessage, ToolExecutionRecord]:
        """Run one tool call; `ToolCallExecutor.execute_single_tool` holds the path."""
        return await self._tool_call_executor.execute_single_tool(
            tc, tool_ctx, stream_callback=stream_callback, advertised=advertised
        )

    def _ingest_tool_message(self, msg: ChatMessage, *, readable: bool) -> ChatMessage:
        """`msg` as the history keeps it: whole under the result cap, else an excerpt.

        The full text goes to this session's artifact directory, which deleting or
        resetting the session removes. `readable` says whether this turn offers
        `tool_result_read`; when it does not, the excerpt says the rest cannot be read
        rather than naming a tool the model cannot call.
        """
        if msg.role != MessageRole.TOOL or msg.content is None:
            return msg
        workspace = self._resolve_workspace_root()
        content = ingest_tool_text(
            msg.content,
            artifacts_dir=artifacts_dir_for(workspace) if workspace is not None else None,
            session_id=self._context.session_id,
            readable=readable,
            workspace_root=workspace,
        )
        if content == msg.content:
            return msg
        # The form is recorded here, where the result was cut, not read back from its text
        # later (#1854).
        return msg.model_copy(update={"content": content, "form": "excerpt"})

    def _window_model(self) -> str | None:
        """The model the next request is sent to, as the window is looked up for it."""
        named = named_model(self._config.llm_config.model_name) or named_model(
            getattr(self._llm, "model", None)
        )
        if named is not None:
            return named
        # A local server's window is per loaded model, so the model the connector resolves
        # for a request naming none is the one to ask about (#1447's `_default_model`).
        # Hosted providers keep the table lookup they had, keyed by the configured name.
        if getattr(self._llm, "provider_name", None) in SERVED_WINDOW_PROVIDERS:
            return named_model(getattr(self._llm, "_default_model", None))
        return None

    def context_window(self) -> int | None:
        """The window this agent's requests are counted against, or `None` when unknown.

        Public for a caller sizing what it hands the agent, such as a room's span budget
        (#1641). It is the figure `_context_window` resolves, so it agrees with the
        compaction trigger and the step budget.
        """
        return self._context_window()

    def _context_window(self) -> int | None:
        """The window the compaction trigger and the step budget count against (#1372).

        The one resolution all three limit sites use -- `should_compact_session` and
        `_request_over_threshold` on `CompactionDriver` (`agent/compaction_driver.py`), and
        `TurnExecutor.fit_step_to_window` (`agent/turn_executor.py`) -- so they cannot
        disagree.
        For a locally served model it is the configured `context_limit`, which the
        connector sends as `num_ctx`, or the window the daemon reported; never the model
        table. `None` when unknown. See `compaction_window`.
        """
        llm = self._llm
        provider = getattr(llm, "provider_name", None)
        base_url = getattr(llm, "base_url", None)
        # The store the connector records the daemon's figures in, so the reader and the
        # writer are one store even when the connector was given its own.
        store = getattr(llm, "context_windows", None)
        return compaction_window(
            provider if isinstance(provider, str) else None,
            self._window_model(),
            base_url=base_url if isinstance(base_url, str) else None,
            configured=self._config.llm_config.context_limit,
            store=store if isinstance(store, OllamaContextWindows) else None,
        )

    def _explicit_threshold(self, window: int) -> int | None:
        """The configured `compaction_threshold_tokens`, when set and below `window`.

        `60_000` is the field's default and so is not an explicit setting. A threshold at
        or above the window would never fire before the server cuts the request, so the
        window's own ratio applies instead (#1372).
        """
        threshold = self._config.llm_config.compaction_threshold_tokens
        if threshold == 60_000 or threshold <= 0 or threshold >= window:
            return None
        return threshold

    async def _observe_context_window(self) -> None:
        """Ask a local server which window it serves, when the connector can (#1372).

        Before each compaction check, so the check counts against the daemon's figure as
        soon as the daemon has one. The connector reads the server only when it holds no
        figure for the model or sent a new `num_ctx` since; a failure leaves the window
        unknown and is never raised into the turn.
        """
        observe = getattr(self._llm, "observe_context_window", None)
        if observe is None or not callable(observe):
            return
        reader = cast(Callable[[str | None], Awaitable[object]], observe)
        try:
            await reader(self._window_model())
        except Exception as exc:  # a window reading must not break a turn
            logger.debug("Could not read the served context window: %s", exc)

    def _reply_reserve(self) -> int:
        """The room a request keeps for the reply, in tokens (#1509).

        The agent's `max_tokens` when it sets one, else `STEP_REPLY_RESERVE_TOKENS`. The
        served window counts the reply as well as the request, so a request fitted to
        the window's last token leaves the model no room to answer.
        """
        max_tokens = self._config.llm_config.max_tokens
        return max_tokens if max_tokens is not None else STEP_REPLY_RESERVE_TOKENS

    def _fit_step_to_window(
        self,
        step_results: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        extra_sections: Sequence[str],
        *,
        readable: bool,
    ) -> str | None:
        """Cut the step that just ran to fit the window, or say why it cannot (#1480).

        See `TurnExecutor.fit_step_to_window`. The turn calls this back through the agent.
        """
        return self._turn_executor.fit_step_to_window(
            step_results, tools, extra_sections, readable=readable
        )

    async def _execute_tools(
        self,
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest],
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        advertised: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord]]:
        """Execute a step's tool calls concurrently; see `TurnExecutor.execute_tools`."""
        return await self._turn_executor.execute_tools(
            tool_calls, stream_callback=stream_callback, advertised=advertised
        )

    def _after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> None:
        """Run the lifecycle hooks over a finished step and keep what they moved (#1732).

        Of the context the hooks return, only `story_id` is kept: the next step's
        `ToolContext` is rebuilt from the turn's fields, and `story_id` is the one this
        writes back. Any other field a hook changes is dropped. A story opened by this
        step is the one the next step's tools see (#1555) -- when the story hook is
        composed in.
        """
        for hook in self._lifecycle_hooks:
            context = hook.after_tool_step(records, context)
        self._turn_story_id = context.story_id

    def advertised_tool_definitions(self) -> list[ToolDefinition]:
        """`available_tools` as the model is sent them, before host binding (`ToolInvoker`)."""
        return self._tool_invoker.advertised_tool_definitions()

    def available_tools(self) -> list[ToolProtocol]:
        """What a turn offers the model, before host binding (`ToolInvoker.available_tools`)."""
        return self._tool_invoker.available_tools()

    def held_tools(self) -> list[ToolProtocol]:
        """The tools this agent actually has, in any conversation (`ToolInvoker.held_tools`)."""
        return self._tool_invoker.held_tools()

    def _may_call_peers(self) -> bool:
        """Whether `a2a_call` could reach anyone: a transport, and a persona naming peers."""
        if self._a2a_transport is None:
            return False
        persona = self._resolved_persona()
        return persona is not None and bool(persona.a2a_peers)

    def memory_share_refusal(self) -> str | None:
        """Why this agent cannot let a sub-agent read its memory, or `None` when it can.

        A child never holds a tool its parent lacks, so sharing needs the parent to have a
        store and to be permitted `query_memory_facts` itself. Worded for the model that
        asked to share, which reads it in the `delegate_subagent` result.
        """
        if self._memory is None:
            return "you have no memory of your own to share"
        allowed = self._config.allowed_tools
        if allowed and QueryMemoryFactsTool.name not in allowed:
            return "you are not allowed to read your own memory, so you cannot share it"
        return None

    async def execute_tool_call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionRecord:
        """Execute one tool call through the agent's own tool path, with no LLM in the loop.

        Same workspace resolution, isolation clamp, path validation and hooks a tool call
        gets when a model asks for it -- which is the point of exposing it. A caller that
        wants to know whether this agent's tools work has to ask along the path the agent
        actually uses; a caller that builds its own `ToolContext` is asking a different
        question and will get `success` from an agent whose own tools are unusable.

        Errors are returned, not raised: a tool that fails yields a record with
        `status == "error"`, exactly as it does mid-turn.

        Two ways this is *not* the same as a model-initiated call, and a caller has to know
        both:

        * `config.allowed_tools` is enforced here by *raising*. It is enforced mid-turn too --
          `_execute_single_tool` checks it before the registry lookup, so a call the model
          invented for a name it was never advertised does not run -- but there the outcome
          is an error *record*, because a turn continues and a raise would end it. Same rule,
          two shapes. `PermissionError` is raised here, distinct from the `KeyError` for a
          name that is not registered at all: "not allowed" and "not there" are different
          answers and neither entry point may conflate them.
        * It leaves **no session audit trail**. No `TOOL_CALL` or `TOOL_RESULT` durable
          event is emitted and no `CALLING_TOOL` state transition happens, because neither
          belongs to a turn. Hooks and path validation still run -- containment is intact --
          but a reader reconstructing the session from its events will not see this call.
        """
        if self._tools is None:
            raise RuntimeError(f"Agent '{self.agent_id}' has no tool registry configured")
        # Existence first, permission second: `PermissionError` means "registered, and
        # withheld from you", which is only answerable once existence is settled. The
        # ordering is load-bearing and is why `ScopedToolRegistry.get` must not report a
        # scoped-out tool as absent -- see its own docstring.
        if self._tool_invoker.resolve(name) is None:
            raise KeyError(f"Tool '{name}' is not registered on agent '{self.agent_id}'")
        allowed = self._config.allowed_tools
        if allowed and name not in allowed:
            raise PermissionError(
                f"Tool '{name}' is registered on agent '{self.agent_id}' but is not in its "
                f"allowed_tools {tuple(allowed)!r}"
            )
        refusal = self._capability_refusal(self._tool_invoker.resolve(name))
        if refusal is not None:
            raise PermissionError(refusal)

        request = ToolCallRequest(
            id=f"direct_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=dict(arguments or {}),
        )
        _messages, records = await self._execute_tools([request])
        return records[0]

    def _subagent_host_fields(
        self, tools: ToolRegistryProtocol | None, ontology: OntologyEngineProtocol | None
    ) -> dict[str, Any]:
        """The `HostDependencies` fields a sub-agent is built with, by field name (#1449).

        Every field of `HostDependencies` is either here or in
        `SUBAGENT_EXCLUDED_HOST_FIELDS`, and a unit test holds the two to exactly the
        dataclass's fields: a field added to the host fails that test until someone decides
        whether a child gets it. Values come from this agent rather than from its host
        object, because what the parent actually runs with lives here -- its hooks include
        those from its config, and an agent built directly has only a fallback host.
        """
        return {
            "bus": self._bus,
            "llm": self._llm,
            "tools": tools or self._tools,
            "tracer": self._tracer,
            "store": self._store,
            "ontology": ontology,
            "skills": self._skills,
            "budget": _ParentSessionBudget(self._budget, self._context.session_id)
            if self._budget is not None
            else None,
            "compactor": self._injected_compactor,
            "hooks": self._hook_runner.hooks,
            "semantic_router": self._semantic_router,
            "tool_binder": self._tool_invoker.binder,
            "plan_generator": self._plan_generator,
            "workspace": self._host.workspace,
            "sandbox": self._host.sandbox,
            "isolation_floor": self._host.isolation_floor,
            "available_isolation": self._host.available_isolation,
            "approvals_answered": self._approvals_answered,
            "lifecycle_hooks": self._lifecycle_hooks,
        }

    async def spawn_subagent(
        self,
        role: str,
        goal: str,
        tools: ToolRegistryProtocol | None = None,
        system_prompt: str | None = None,
        ontology: OntologyEngineProtocol | None = None,
        share_parent_memory: bool = False,
    ) -> BaseAgent:
        """Spawn a specialized sub-agent dynamically with strict context isolation (P2, P4).

        Raises `PermissionError` when this agent's sub-agents are switched off
        (`subagent_tools_enabled`, #1167). Refusing here and not only at the
        `delegate_subagent` tool is what covers every way in: the tool, a direct caller,
        and a tool nobody declared as one that starts agents.

        A sub-agent is new and throwaway, so it gets no memory tools (#1431). With
        `share_parent_memory`, it gets `query_memory_facts` alone, bound read-only to this
        agent's store -- never record or retract, since the store rewrites its whole
        document with no lock and a second writer would overwrite this agent's saves. When
        `memory_share_refusal` gives a reason, the flag grants nothing.
        """
        if not self.subagent_tools_enabled:
            raise PermissionError(
                f"Agent '{self.agent_id}' may not start a sub-agent: enable_subagent_tools "
                "is off (turn on 'Allow sub-agents' in the persona's settings to allow it)"
            )
        subagent_id = f"{self.agent_id}_sub_{int(asyncio.get_running_loop().time() * 1000)}"
        prompt = system_prompt or f"You are a specialized sub-agent for role '{role}'. Goal: {goal}"

        # P4: Context isolation - sub-agents do not inherit parent chat history
        #
        # A child never holds more tools than its parent. The parent's *resolved* list is
        # read here, at spawn time, so a persona edited after the parent was created governs
        # the next child too, and, less the memory tools (#1431), it becomes the child's own
        # `allowed_tools` so the child refuses a call outside it rather than merely not
        # being offered the tool. `query_memory_facts` stays only when this agent shares its
        # memory. A persona agent's registry used to be wrapped in a scope proxy the child
        # inherited; since the persona's list is resolved inside the agent (#892) there is
        # no proxy to inherit.
        shared_memory = (
            self._memory if share_parent_memory and self.memory_share_refusal() is None else None
        )
        child_allowed = tuple(
            name
            for name in self._config.allowed_tools
            if name not in BASE_MEMORY_TOOLS
            or (shared_memory is not None and name == QueryMemoryFactsTool.name)
        )
        child_tools = tools or self._tools
        # An empty list permits everything, so a parent permitted only memory tools must not
        # hand its child an empty list over the full registry. Such a child is permitted
        # nothing, so it gets an empty registry and no skills: with skills it would register
        # `load_skill` itself, and its empty list would let it run it.
        permits_nothing = bool(self._config.allowed_tools) and not child_allowed
        if permits_nothing:
            child_tools = ToolRegistry()
        sub_config = AgentConfig(
            agent_id=subagent_id,
            name=f"{self._config.name}_{role}",
            role=role,
            system_prompt=prompt,
            allowed_tools=child_allowed,
            # The parent's *effective* flags, persona included, as they stand at spawn: a
            # child holds no capability its parent was refused (#1167). The child has no
            # persona of its own, so its config is the whole of what it is allowed.
            enable_write_tools=self.write_tools_enabled,
            enable_subagent_tools=self.subagent_tools_enabled,
            llm_config=self._config.llm_config,
            workspace_dir=self._config.workspace_dir,
            read_roots=self._config.read_roots,  # a child reads what its parent reads
            isolation=self._config.isolation,
        )

        # A session of its own, always: the parent's would put two writers on one
        # `SessionState` (G3). The option to share it was removed with #1449.
        sub_context = AgentContext(
            session_id=f"sess_{subagent_id}",
            agent_id=subagent_id,
            parent_agent_id=self.agent_id,
            depth=self._context.depth + 1,
            workspace_root=self._context.workspace_root,
        )

        # Built by the same mapping as every other agent (`compose_agent`'s construction
        # half), from a host derived from this one (#1449).
        # Enforcement is inherited -- the parent's hooks, its budget, its tool binding and
        # its host's isolation floor and sandbox -- so a child cannot run a tool its parent
        # would have had approved or refused, nor spend past the parent's ceiling.
        # Capabilities are inherited too. What is not: persona and cross-session memory
        # (the child is ephemeral and has no persona of its own, see above), and the
        # ontology unless the caller passes one.
        #
        # The hooks are the parent runner's *current* list, registered in a runner of the
        # child's own rather than the parent's runner itself: a runner publishes through
        # its agent's publisher, which stamps the sender, so sharing it would report the
        # child's hook decisions as the parent's.
        from uclone_x.agent.composition import HostDependencies, build_agent

        host_fields = self._subagent_host_fields(child_tools, ontology)
        if permits_nothing:
            host_fields["skills"] = None
        child_host = HostDependencies(**host_fields)
        sub_agent = build_agent(sub_config, child_host, sub_context)
        # Bound after construction, not given to the host as `memory=`: that would hand the
        # child the whole store and with it record and retract (#1431).
        if shared_memory is not None:
            reader = QueryMemoryFactsTool(ReadOnlyMemory(shared_memory))
            sub_agent._tool_invoker.local_tools[reader.name] = reader
            if sub_agent._tools is not None and sub_agent._tools.get(reader.name) is None:
                sub_agent._tools.register(reader)

        if self._bus is not None:
            spawn_event = AgentEvent(
                type=EventType.SUBAGENT_SPAWN,
                recipient_id=subagent_id,
                topic=f"swarm.subagent.{subagent_id}",
                payload={"role": role, "goal": goal, "isolated": "True"},
            )
            if self._publisher is not None:
                await self._publisher.publish(spawn_event)
            else:
                await self._bus.publish(spawn_event)

        return sub_agent

    @staticmethod
    def _extract_child_steps(subagent: Any) -> int:
        """Extract steps taken by a subagent from run_steps or turn_counter."""
        steps = 0
        for attr in ("run_steps", "turn_counter", "_turn_counter"):
            val = getattr(subagent, attr, None)
            if isinstance(val, int) and val > steps:
                steps = val
        return steps

    async def delegate_task(
        self,
        subagent: BaseAgent,
        task_prompt: str,
    ) -> TurnResult:
        """Delegate a task to a sub-agent concurrently and collect turn result (P2, P4)."""
        await subagent.start()
        try:
            result = await subagent.execute_turn(task_prompt)
            if self._bus is not None:
                done_payload: dict[str, Any] = {
                    "content": result.content,
                    "is_completed": str(result.is_completed),
                }
                if result.error is not None:
                    done_payload["error"] = result.error
                if result.tool_executions:
                    done_payload["tool_executions"] = [
                        {
                            "tool_call_id": te.tool_call_id,
                            "tool_name": te.tool_name,
                            "arguments": cast(dict[str, Any], unwrap_immutable(te.arguments)),
                            "output": unwrap_immutable(te.output),
                            "status": te.status,
                            "error": te.error,
                            "duration_ms": te.duration_ms,
                        }
                        for te in result.tool_executions
                    ]
                done_event = AgentEvent(
                    type=EventType.SUBAGENT_DONE,
                    recipient_id=self.agent_id,
                    topic=f"swarm.subagent.{subagent.agent_id}",
                    payload=done_payload,
                    # This event carries the sub-agent's result, so it is result-bearing
                    # and P6 applies to it exactly as to an AGENT_REPLY.
                    provenance=require_provenance(result.provenance, "TurnResult"),
                )
                if subagent._publisher is not None:
                    await subagent._publisher.publish(done_event)
                elif self._publisher is not None:
                    await self._publisher.publish(done_event)
                else:
                    await self._bus.publish(done_event)
            return result
        finally:
            child_steps = self._extract_child_steps(subagent)
            if child_steps > 0 and getattr(subagent, "_steps_deducted", None) is not True:
                self.consume_steps(child_steps)
                subagent._steps_deducted = True
            await subagent.stop()
