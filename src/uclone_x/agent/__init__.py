"""Agent subsystem: BaseAgent, reactive state machine, and subagent swarm management."""

from uclone_x.agent.base import VALID_TRANSITIONS, BaseAgent
from uclone_x.agent.composition import HostDependencies, MissingCapabilityError, compose_agent
from uclone_x.agent.hooks import (
    BaseHook,
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
    HookRunner,
    ScriptHook,
)
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
    FsScope,
    ModelTier,
    PersonaDefinition,
    PlanState,
    PlanStep,
    SubagentInvocation,
    SubAgentSpec,
    ToolExecutionRecord,
    TurnResult,
)
from uclone_x.agent.persona_registry import (
    PersonaRegistry,
    get_default_persona_registry,
)
from uclone_x.agent.protocols import (
    BaseAgentProtocol,
    SubAgentSupervisorProtocol,
)
from uclone_x.agent.session import (
    DEFAULT_SESSION_STORAGE_DIR,
    CompactionResult,
    SessionState,
    SessionStore,
    resolve_session_path,
    validate_session_id,
    verify_record_identity,
)
from uclone_x.errors import StepBudgetExceededError

__all__ = [
    "AgentConfig",
    "AgentContext",
    "AgentLLMConfig",
    "AgentState",
    "BaseAgent",
    "BaseAgentProtocol",
    "BaseHook",
    "compose_agent",
    "CompactionResult",
    "DEFAULT_SESSION_STORAGE_DIR",
    "FailurePolicy",
    "FsScope",
    "HookAction",
    "HookContext",
    "HookDecision",
    "HookEvent",
    "HookRunner",
    "HostDependencies",
    "MissingCapabilityError",
    "ModelTier",
    "PersonaDefinition",
    "PersonaRegistry",
    "get_default_persona_registry",
    "PlanState",
    "PlanStep",
    "ScriptHook",
    "SessionState",
    "SessionStore",
    "SubagentInvocation",
    "SubAgentSpec",
    "SubAgentSupervisorProtocol",
    "ToolExecutionRecord",
    "StepBudgetExceededError",
    "TurnResult",
    "VALID_TRANSITIONS",
    "validate_session_id",
    "resolve_session_path",
    "verify_record_identity",
]
