"""Data models and state definitions for UClone-X agents."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.core.immutable import ImmutableJsonMapping, ImmutableStrMapping

# Lowered to `uclone_x.core.models` (#1734) and re-exported here under their old spelling.
from uclone_x.core.models import BASE_MEMORY_TOOLS as BASE_MEMORY_TOOLS
from uclone_x.core.models import BASE_PERSONA_TOOLS as BASE_PERSONA_TOOLS
from uclone_x.core.models import BASE_SELF_TOOLS as BASE_SELF_TOOLS
from uclone_x.core.models import RETIRED_MODEL_TIERS as RETIRED_MODEL_TIERS
from uclone_x.core.models import AgentLLMConfig as AgentLLMConfig
from uclone_x.core.models import ModelTier as ModelTier
from uclone_x.core.models import PersonaDefinition as PersonaDefinition
from uclone_x.core.models import PlanState as PlanState
from uclone_x.core.models import PlanStep as PlanStep
from uclone_x.core.provenance import Provenance
from uclone_x.core.tool_results import StepRefusalCode
from uclone_x.errors import ProviderFailureError, ProviderFailureKind
from uclone_x.llm.models import TokenUsage, ToolCallRequest
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
    max_subagent_depth: int = 2
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
    tools_module: str | None = Field(
        default=None,
        description=(
            "How the agent builds its tools layer (#2188): `native`, `pinned` or `bound`. "
            "`None` takes the provider's default (`select_tools_module`). Read once, when "
            "the agent is built; a name this build does not have is refused then."
        ),
    )
    approval_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds when waiting for human approval of a tool call.",
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
    opens_story: bool = Field(
        default=False,
        description="The executed tool's own `opens_story` declaration (#1555), copied at "
        "the same site and only for a call that succeeded. What lets the runtime read the "
        "conversation's new story from an output without trusting an output's shape.",
    )
    artifacts: tuple[str, ...] = Field(
        default=(),
        description="The workspace files the call says it produced, from the tool's own "
        "`ToolResult.artifacts`, copied at the one site that ran it and only for a call that "
        "succeeded (#2085). Read through `produced_paths`, never from the output's shape: "
        "the image result names its pictures by link only (#2013) and the image tool does "
        "not declare `writes_files` (#2079), and each of those broke a reader that parsed.",
    )

    @field_validator("artifacts")
    @classmethod
    def _workspace_relative(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Each path once, in order, with `/` separators; an absolute path is dropped.

        Here and not on `ToolResult`: the image tools set `artifacts` with `model_copy`,
        which skips that model's validators. An absolute path names no workspace file, and
        stripping its root would name a different one.
        """
        kept: list[str] = []
        for raw in value:
            path = raw.strip().replace("\\", "/")
            if path and not path.startswith("/") and ":" not in path[:3] and path not in kept:
                kept.append(path)
        return tuple(kept)

    @property
    def produced_paths(self) -> tuple[str, ...]:
        """The workspace files this call produced: the one answer every reader uses (#2085).

        Nothing for a call that did not succeed. Otherwise the tool's declared `artifacts`;
        only when it declared none, and only for a tool that declares `writes_files`, the
        `path` and `paths` its output names -- the shape external MCP tools, peers and the
        story tools still report by. `file_read` returns a `path` too and declares no
        writes, so a file it read is never counted as produced.
        """
        if self.status is not ToolResultStatus.SUCCESS:
            return ()
        if self.artifacts:
            return self.artifacts
        if not self.writes_files or not isinstance(self.output, dict):
            return ()
        candidates: list[object] = [self.output.get("path")]
        listed = self.output.get("paths")
        if isinstance(listed, list):
            candidates.extend(listed)
        named: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate and candidate not in named:
                named.append(candidate)
        return tuple(named)


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
    "usage_limit",
    "provider_timeout",
    "model_without_tools",
    "model_unavailable",
    "provider_auth",
    "provider_quota",
    "provider_unreachable",
    "provider_outage",
    "provider_error",
    "persona_edit_failed",
    "tool_call_unreadable",
    "cancelled",
]
"""How a turn's step run ended: the vocabulary of `TURN_END.stop_reason` and `TurnResult`.

`cancelled` reaches only `TURN_END`, because a cancelled turn raises rather than returning.

`step_results_over_window` refuses a step whose tool results, together, cannot fit the
context window even as excerpts (#1480). Nothing over the window was sent. It is not a
refusal a retry meets again: the next attempt may ask for fewer things at once.

`usage_limit` is the user's own limit on paid-model tokens (`llm-token-gateway.md` §4.3),
refused before the call was sent. Its `error` is a plain sentence naming the window, when
it lifts and the two remedies, written to be shown as is. It is a refusal until the window
lifts or the limit is raised -- and unlike `budget_exceeded`, a new conversation does not
cure it, because the limit is system-wide.

`provider_timeout` is the one failure ending that is named rather than left to `error`.
Every other provider failure arrives as a string in `error` and is indistinguishable from
the next one, which is exactly what #1277 cost: a `frontier_live` probe cut off at the
connector's ceiling and a probe whose host was not running both reached the report as
`UNREACHABLE` with a message, and telling them apart meant reading the duration
distribution. A consumer that must not count a cut-off turn as the model's failure needs
a field it can test, not a phrase it can grep. It is not a refusal -- `turn_refusal`
leaves it `None` -- because a retry at a longer ceiling is exactly the right response.

`model_without_tools` is a provider's refusal of the chosen model because it cannot take
tool definitions, which every clone turn sends. Its `error` is a plain sentence naming the
model and the remedy, written to be shown as is. It is a refusal: a retry on the same model
is refused the same way, so `turn_refusal` maps it to `RoomTurnRefusal.MODEL_WITHOUT_TOOLS`.

`model_unavailable`, `provider_auth`, `provider_quota`, `provider_unreachable`,
`provider_outage` and `provider_error` are a hosted provider's failure, one per
`ProviderFailureKind` and spelled the same (#1630). The turn's `provider_failure` carries
the plain sentence for it. The first two are refusals -- a retired model and a rejected key
fail every retry until the setting changes -- and the rest are not.

`persona_edit_failed` is a saved persona edit this seat could not apply at the turn's start
(#1904). The turn did not run, the seat kept its previous definition, and the edit was taken
off the stage, so it is not tried again. Its `error` is a fixed plain sentence, never the
cause's text: the cause is in the log. It is not a refusal -- a retry runs, under the
previous definition.

`tool_call_unreadable` is a ``k_act`` reply holding a ``<tool_call>`` block that does not
read as a call (#2188, owner ruling 2026-10-04). Nothing of that reply ran or entered
history, and the turn did not fall back to native tool calling. Its `error` is a fixed
plain sentence; the reply is in the log's `MODEL_RESPONSE` event. It is not a refusal -- a
retry samples a new reply.
"""


class ProviderFailure(BaseModel):
    """A hosted provider's failure, as a head shows it (#1630).

    `message` is the `ProviderFailureError`'s own sentence: it says what stopped and whose
    side the cause is on, and carries no status, body, URL or class name, so a head shows it
    as is and adds only where to act. It is carried apart from `TurnResult.error` so no head
    has to decide whether an `error` string is safe to show.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: ProviderFailureKind
    message: str
    retryable: bool
    provider: str | None = Field(
        default=None,
        description="The provider's display name, e.g. 'Google', so a head can name where "
        "that provider's key is set. None on a record stored before it was carried.",
    )
    clone: str | None = Field(
        default=None,
        description="The clone whose own model failed, when the failure is that model's "
        "(model-gateway §3.6). None when the clone follows the system default.",
    )
    model_ref: str | None = Field(
        default=None,
        description="The clone's own model ref that could not be used, e.g. 'gpu-box/qwen3'.",
    )
    action: Literal["use_system_default"] | None = Field(
        default=None,
        description="The one action a head offers: 'use_system_default' clears the clone's "
        "own model so it follows Settings. The turn is never re-run on the default unasked.",
    )

    @classmethod
    def of(cls, exc: ProviderFailureError) -> ProviderFailure:
        """The record of a raised provider failure."""
        return cls(kind=exc.kind, message=str(exc), retryable=exc.retryable, provider=exc.provider)


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
    story_id: str | None = Field(
        default=None,
        description="The story the turn left open, as its lifecycle hooks moved it between "
        "steps (#1732) -- the `story_id` it was called with when no hook moved it, or when "
        "no hook that moves stories is composed in. A room keeps this as its open story "
        "(#1775), including on a turn that failed after a step had moved it.",
    )
    error: str | None = None
    error_code: StepRefusalCode | None = Field(
        default=None,
        description="Set, beside `error`, when the turn refused a step (#1862): the key a "
        "head translates `error` by, so a head writing in the person's language says the "
        "refusal from its own catalog. `error` stays Core's English sentence. `None` on "
        "every other turn.",
    )
    provider_failure: ProviderFailure | None = Field(
        default=None,
        description="Set, beside `error`, when the turn failed on a hosted provider's "
        "failure the connector could classify (#1630). `None` on every other turn.",
    )
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
        "refused the turn, which a retry meets again until the ceiling changes (#969); "
        "`usage_limit` is the user's system-wide paid-model limit, whose `error` is "
        "written to be shown as is. "
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
