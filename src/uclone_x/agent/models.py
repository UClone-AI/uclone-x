"""Data models and state definitions for UClone-X agents."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.core.immutable import ImmutableJsonMapping, ImmutableStrMapping
from uclone_x.core.provenance import Provenance
from uclone_x.llm.models import TokenBudget, TokenUsage, ToolCallRequest
from uclone_x.sandbox.models import (
    IsolationPolicy,
    WorkspaceIsolation,
)
from uclone_x.tools.models import ToolResultStatus


class AgentState(StrEnum):
    """6-stage reactive lifecycle states defined in FR-1 plus terminal states."""

    IDLE = "IDLE"
    INGESTING = "INGESTING"
    REASONING = "REASONING"
    CALLING_TOOL = "CALLING_TOOL"
    AWAITING_INPUT = "AWAITING_INPUT"
    EMITTING_RESPONSE = "EMITTING_RESPONSE"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"


class ModelTier(StrEnum):
    """Model specialization tiers for agents and sub-agents."""

    INHERIT = "inherit"
    FAST = "fast"
    PRO = "pro"
    FLASH_LITE = "flash_lite"
    CUSTOM = "custom"


class FsScope(StrEnum):
    """Filesystem scope a sub-agent runs in.

    Named `fs_scope` per issue 2026-09-02-019, which separated this from the sandbox
    security axis (`IsolationLevel`) after `mode` came to name both. This axis says
    which files a sub-agent shares with its parent; it is not a security boundary.

    The former `BRANCH` member is gone: a git branch is a Builder-layer concept, and
    AGENTS.md's Builder-versus-`ucx agent` separation forbids Builder metadata from
    leaking into the runtime abstractions.
    """

    INHERIT = "inherit"
    ISOLATED = "isolated"
    SHARED = "shared"


class AgentLLMConfig(BaseModel):
    """Per-agent LLM parameter and token budget configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model_tier: ModelTier = ModelTier.INHERIT
    model_name: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    token_budget: TokenBudget | None = None
    auto_compact: bool = True
    compaction_threshold_tokens: int = 60_000
    context_limit: int | None = Field(
        default=None,
        description="Explicit maximum context window tokens for this agent's model. "
        "When provided or resolved from model_name, auto-compaction triggers at 70% of this limit.",
    )


#: The tools every persona is given, whatever its own `allowed_tools` lists (#1402). Owner
#: ruling, 2026-09-23: an agent is given the tools it basically needs by default. Before
#: this only `clone` -- the one built-in with no list, and so no restriction -- could save
#: a memory; every other persona's call to `record_memory_fact` was refused.
#:
#: The rule for membership: a tool belongs here only if it has no effect outside the
#: agent's own memory. So the three memory tools (bound to the agent's own store, see
#: `AGENT_BOUND_TOOL_TYPES`) and the three read-only workspace tools, which
#: `writes_files = False` declares and the workspace boundary confines, plus
#: `tool_result_read` (#1422), which reads back this conversation's own shortened tool
#: results and nothing else -- without it, an excerpt would name a reader the persona
#: cannot call. Anything that writes a file, runs a command, installs, reaches the
#: network or makes an image stays in the persona's own list.
#:
#: Names, not classes: this module sits below the tool packages. A test holds each name to
#: the class that registers under it, so a rename cannot leave a dead entry here.
#:
#: Naming a tool here grants permission; it does not register anything. An agent composed
#: with no memory store still has no memory tools, and is not offered one it cannot run.
#:
#: The memory part is named on its own, `BASE_MEMORY_TOOLS`, because a sub-agent is not
#: given it (#1431): `spawn_subagent` removes these names from the list a child inherits.
BASE_MEMORY_TOOLS: tuple[str, ...] = (
    "record_memory_fact",
    "query_memory_facts",
    "retract_memory_fact",
)

BASE_PERSONA_TOOLS: tuple[str, ...] = (
    *BASE_MEMORY_TOOLS,
    "file_read",
    "file_search",
    "directory_list",
    "tool_result_read",
)


class PersonaDefinition(BaseModel):
    """Configuration schema for dynamically defined sub-agent personas."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(description="Unique name identifier for the dynamic agent persona")
    role: str = Field(description="Human-readable role (e.g. 'Security Reviewer')")
    description: str = Field(default="", description="Description of what this persona does")
    system_prompt: str = Field(description="System instructions and persona constraints")
    allowed_tools: tuple[str, ...] = Field(
        default_factory=tuple, description="Authorized tool names"
    )
    llm_config: AgentLLMConfig = Field(
        default_factory=AgentLLMConfig, description="Dedicated LLM parameters"
    )
    enable_write_tools: bool = Field(
        default=False, description="Whether persona has write permissions"
    )
    enable_subagent_tools: bool = Field(
        default=False, description="Controls recursive sub-agent creation"
    )

    @property
    def granted_tools(self) -> tuple[str, ...]:
        """The tools this persona may use: its own `allowed_tools` plus `BASE_PERSONA_TOOLS`.

        This, not `allowed_tools`, is what every site that scopes an agent to a persona
        reads. `allowed_tools` stays the persona's own list -- what its file says and what
        its editor shows and saves -- so the base set is never written into a persona.

        An empty `allowed_tools` is no restriction at all (every registered tool), and
        stays empty here: adding the base set to it would turn "everything" into "only
        the base set".

        The persona's own order is kept and the base names follow, each once.
        """
        if not self.allowed_tools:
            return ()
        own = self.allowed_tools
        return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)


class SubagentInvocation(BaseModel):
    """Parameters to invoke a configured dynamic sub-agent."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type_name: str = Field(description="Name of the registered PersonaDefinition")
    role: str = Field(description="Role description for this invocation")
    prompt: str = Field(description="Task prompt for the subagent")
    fs_scope: FsScope = Field(
        default=FsScope.INHERIT, description="Filesystem scope shared with the parent"
    )
    timeout_seconds: float = Field(default=300.0, description="Turn execution timeout")
    llm_override: AgentLLMConfig | None = Field(
        default=None, description="Optional runtime LLM parameter override"
    )


#: The default system prompt, assembled from the components in `uclone_x.agent.prompts`
#: for an unspecified model family. `BaseAgent.effective_system_prompt` re-frames the
#: steerability section once the active model is known (#871).
DEFAULT_SYSTEM_PROMPT = compose_system_prompt()


class AgentConfig(BaseModel):
    """Configuration blueprint for initializing an agent."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, arbitrary_types_allowed=True
    )

    agent_id: str
    name: str
    role: str = ""
    description: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    seat_framing: str = Field(
        default="",
        description=(
            "Text that places the agent in a shared conversation, set by the room for a "
            "seat. When present it leads the identity prompt, ahead of the persona's "
            "instructions or `system_prompt`. The agent composes the two itself, so the "
            "anchor it stores and the system message it sends are built by one function."
        ),
    )
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    llm_config: AgentLLMConfig = Field(
        default_factory=AgentLLMConfig,
        description="The single source of truth for model tier and sampling. The former "
        "sibling `model_tier: str` and `temperature: float` fields are gone: they "
        "duplicated fields inside this object, one of them untyped, with no rule for "
        "which won (issue 2026-09-02-036).",
    )
    require_evidence_before_answer: bool = Field(
        default=False,
        description=(
            "Whether a turn that produced no tool execution is sent back once for "
            "evidence before its answer is accepted. Off by default: a greeting answered "
            "without tools is correct. On for agents whose answers are claims about "
            "something they can inspect -- see #697, where a median of one tool call met "
            "a median declared horizon of six."
        ),
    )
    max_steps: int = Field(
        default=50,
        description="Maximum agent execution steps (tool rounds) allowed within a single turn per P4.",
    )
    max_turns: int = Field(
        default=50,
        json_schema_extra={"deprecated": True},
        description="[Deprecated alias for max_steps] Maximum steps allowed within a single turn.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_step_budget(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            raw = dict(cast(Mapping[str, Any], data))
            steps = raw.get("max_steps")
            turns = raw.get("max_turns")
            if steps is not None and turns is not None:
                if steps != turns:  # AgentConfig budget conflict check
                    raise ValueError(
                        f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                    )
            elif turns is not None and steps is None:
                raw["max_steps"] = turns
            elif steps is not None and turns is None:
                raw["max_turns"] = steps
            return raw
        if hasattr(data, "max_steps") and hasattr(data, "max_turns"):
            steps = data.max_steps
            turns = data.max_turns
            if steps is not None and turns is not None and steps != turns:
                raise ValueError(
                    f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                )
        return data

    max_subagent_depth: int = 2
    max_concurrent_subagents: int = 5
    workspace_dir: str | Path | None = Field(
        default=None,
        description="Workspace root directory for tool execution boundary.",
    )
    read_roots: tuple[Path, ...] = Field(
        default=(),
        description="Folders outside the workspace that read-only file tools may also read.",
    )
    isolation: IsolationPolicy = Field(
        default_factory=WorkspaceIsolation,
        description="Isolation policy for tool executions. Defaults to WorkspaceIsolation per P3.",
    )
    enable_write_tools: bool = True
    enable_subagent_tools: bool = True
    hooks: tuple[BaseHook, ...] = Field(default_factory=tuple)
    persona: str | None = None
    approval_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds when waiting for human approval of a tool call.",
    )


class PlanStep(BaseModel):
    """A discrete step within an interactive agent execution plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    index: int = Field(description="Sequential index of the plan step")
    description: str = Field(description="Description of the action/task for this step")
    completed: bool = Field(default=False, description="Whether this step has been completed")
    verification: str | None = Field(
        default=None, description="Optional verification criteria or outcome"
    )


class PlanState(BaseModel):
    """Interactive chat plan state tracking step progress."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}",
        description="Unique plan identifier",
    )
    title: str = Field(description="Title or goal of the plan")
    steps: tuple[PlanStep, ...] = Field(
        default_factory=tuple, description="Ordered tuple of plan steps"
    )
    status: Literal["proposed", "in_progress", "completed", "rejected"] = Field(
        default="proposed", description="Overall lifecycle status of the plan"
    )


class AgentContext(BaseModel):
    """Runtime context and state for an active agent execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    session_id: str
    agent_id: str
    current_state: AgentState = AgentState.IDLE
    turn_index: int = 0
    trace_id: str | None = None
    parent_agent_id: str | None = None
    depth: int = 0
    workspace_root: Path | None = Field(
        default=None,
        description="Optional runtime workspace boundary override.",
    )
    metadata: ImmutableStrMapping = Field(default_factory=dict)
    current_plan: PlanState | None = Field(
        default=None,
        description="Optional active chat plan state tracking step progress.",
    )


class SubAgentSpec(BaseModel):
    """Specification for dynamically defining and launching a sub-agent."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    role: str
    description: str = ""
    system_prompt: str
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    llm_config: AgentLLMConfig = Field(default_factory=AgentLLMConfig)
    fs_scope: FsScope = FsScope.INHERIT
    workspace_dir: str | Path | None = Field(
        default=None,
        description="Workspace boundary directory for subagent tool executions.",
    )
    isolation: IsolationPolicy = Field(
        default_factory=WorkspaceIsolation,
        description="Isolation policy for subagent tool executions. Defaults to WorkspaceIsolation per P3.",
    )
    max_steps: int = Field(
        default=20,
        description="Maximum agent execution steps allowed within a single turn per P4.",
    )
    max_turns: int = Field(
        default=20,
        json_schema_extra={"deprecated": True},
        description="[Deprecated alias for max_steps] Maximum steps allowed within a single turn.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_step_budget(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            raw = dict(cast(Mapping[str, Any], data))
            steps = raw.get("max_steps")
            turns = raw.get("max_turns")
            if steps is not None and turns is not None:
                if steps != turns:  # SubAgentSpec budget conflict check
                    raise ValueError(
                        f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                    )
            elif turns is not None and steps is None:
                raw["max_steps"] = turns
            elif steps is not None and turns is None:
                raw["max_turns"] = steps
            return raw
        if hasattr(data, "max_steps") and hasattr(data, "max_turns"):
            steps = data.max_steps
            turns = data.max_turns
            if steps is not None and turns is not None and steps != turns:
                raise ValueError(
                    f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                )
        return data

    enable_write_tools: bool = False
    enable_subagent_tools: bool = False


class ToolExecutionRecord(BaseModel):
    """Enriched record of an executed tool invocation during an agent turn."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tool_name: str
    arguments: ImmutableJsonMapping = Field(default_factory=dict)
    output: JsonValue = None
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    error: str | None = None
    duration_ms: float = 0.0
    tool_call_id: str | None = None
    writes_files: bool = Field(
        default=False,
        description="The executed tool's own `writes_files` declaration (#1167), copied "
        "at the one site that ran it. A room reads this to decide whether an output's "
        "`path` names a file the conversation *wrote* (#1354): `file_read` returns a "
        "`path` too, and the shape of an output is not a capability. Set on every record "
        "for a call that reached the tool, whether it succeeded, failed or raised, since a "
        "tool can write before it fails (#1366). False for a call that never reached a tool "
        "(unknown or refused), which cannot have run it.",
    )
    spawns_subagents: bool = Field(
        default=False,
        description="The executed tool's own `spawns_subagents` declaration (#1167), "
        "copied at the same site. What lets a room's topology draw a sub-agent a seat "
        "started (#1355, P4) without matching on a tool's name.",
    )


TurnStopReason = Literal[
    "not_started",
    "blocked_by_hook",
    "model_stopped",
    "model_stopped_after_nudge",
    "model_stopped_after_grounding_nudge",
    "model_stopped_after_both_nudges",
    "model_stopped_after_unproductive_tools",
    "no_tools_registered",
    "step_budget_exceeded",
    "step_results_over_window",
    "budget_exceeded",
    "provider_timeout",
    "cancelled",
]
"""How a turn's step run ended: the vocabulary of `TURN_END.stop_reason` and `TurnResult`.

`cancelled` reaches only `TURN_END`, because a cancelled turn raises rather than returning.

`step_results_over_window` refuses a step whose tool results, together, cannot fit the
context window even as excerpts (#1480). Nothing over the window was sent. It is not a
refusal a retry meets again: the next attempt may ask for fewer things at once.

`provider_timeout` is the one failure ending that is named rather than left to `error`.
Every other provider failure arrives as a string in `error` and is indistinguishable from
the next one, which is exactly what #1277 cost: a `frontier_live` probe cut off at the
connector's ceiling and a probe whose host was not running both reached the report as
`UNREACHABLE` with a message, and telling them apart meant reading the duration
distribution. A consumer that must not count a cut-off turn as the model's failure needs
a field it can test, not a phrase it can grep. It is not a refusal -- `turn_refusal`
leaves it `None` -- because a retry at a longer ceiling is exactly the right response.
"""


class TurnResult(BaseModel):
    """Result of a single agent reasoning turn."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    turn_index: int
    content: str
    tool_calls: tuple[ToolCallRequest, ...] = Field(default_factory=tuple)
    tool_executions: tuple[ToolExecutionRecord, ...] = Field(
        default_factory=tuple,
        description="Every tool call that finished during the turn, across all its steps, "
        "including on a turn that failed or hit a budget after some steps had run tools. "
        "An errored turn's list is the calls that ran before the failure, not `()` (#1366).",
    )
    tool_executions_complete: bool = Field(
        default=True,
        description="False when the turn failed while a step's tools were still running. "
        "Tools in that step may have run -- and written files -- without their records "
        "reaching `tool_executions`, so the list is then a lower bound and must not be read "
        "as everything the turn did (P6, #1366).",
    )
    is_completed: bool = False
    steps_taken: int = Field(
        default=0,
        description=(
            "Model round-trips this turn consumed. Distinct from `len(tool_executions)`: "
            "a step may call several tools or none. Reported because effort is only "
            "readable against what a task required, and #697 had to reconstruct it."
        ),
    )
    unsupported_claims: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Specifics this turn's answer asserts -- paths, filenames, multi-digit "
            "numbers -- that appear in nothing the turn read or was told, in the order "
            "the answer made them. Populated on every completed turn regardless of "
            "`require_evidence_before_answer`, which only decides whether the loop acts "
            "on them. It is an observation, not a verdict: a listed specific may be "
            "correct and merely re-derived, and the support test is a substring search, "
            "so an unlisted one is not certified. See `agent/grounding.py` for what "
            "counts as a specific and why prose is not collected. #697 is the "
            "measurement -- a median of one tool call against a median declared horizon "
            "of six, with the answer naming a default the agent never opened the file to "
            "read. **Empty is not a cleanliness signal.** On a turn that ended at the step "
            "budget (`is_completed=False`) the specifics were never computed at all, so "
            "the tuple is empty by absence rather than by finding nothing; and even on a "
            "completed turn the support test is a substring search over everything read, "
            "so the emptier a turn's tuple the more that turn happened to read. A consumer "
            "must gate on `is_completed` before reading anything into `()`."
        ),
    )
    text_emitted_tool_calls: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Registered tools this turn's message *content* named in tool-call "
        "shape while the structured `tool_calls` channel was empty, one entry per "
        "occurrence, in the order the steps ran. Read it together with "
        "`tool_executions`: non-empty here with `tool_executions` empty is the #694 "
        "failure -- work the model asked for was discarded and the blob it asked with "
        "was returned as the answer, which `qwen2.5-coder:14b` does for every call and "
        "which produced a full 100-problem capability figure for a model whose every tool "
        "call fell on the floor. The detector is conservative but not exact: a correct "
        "answer that quotes a tool's own schema back (`{'name': ..., 'parameters': ...}`) "
        "is call-shaped and is counted, so a non-empty value is evidence to read, not a "
        "proof that anything was lost. Nothing recovers these -- they are names, not "
        "invocations -- so the field exists to make the failure legible where it is "
        "otherwise indistinguishable from a model that chose not to use tools.",
    )
    error: str | None = None
    stop_reason: TurnStopReason | None = Field(
        default=None,
        description="How the step run ended -- the value this turn's `TURN_END` event "
        "records. It says why the loop stopped, not whether the turn succeeded: read "
        "`error` for that. `not_started` is the value for any failure before the loop "
        "decides to stop -- a provider error on any step, after tool calls included -- "
        "and a failure after that decision keeps the decided value, so `model_stopped` "
        "can carry `error` too. `blocked_by_hook` is how a consumer tells a `PRE_TURN` hook "
        "refusal from any other failure; before #970 the CLI compared `content` with the "
        "engine's wording instead. `budget_exceeded` is a token ceiling that "
        "refused the turn, which a retry meets again until the ceiling changes (#969). "
        "`None` only from a `TurnResult` built outside "
        "`BaseAgent.execute_turn`.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Event ID or correlation ID of the causing input event (Issue #60).",
    )
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )
    router_tier: str | None = None
    persona: str | None = None
    active_skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of approved skills available to the agent during this turn.",
    )
    loaded_skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of skills explicitly loaded into session context.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Total tokens consumed across all model calls of this turn. "
        "Summed over every completed step. `count_source` is `provider` only if every "
        "step's was `provider`; otherwise it is the least certain source (e.g. `estimate`). "
        "`None` when no model call completed (e.g. hook block or immediate pre-loop failure).",
    )
