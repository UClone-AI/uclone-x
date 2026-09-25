"""Concrete BaseAgent implementing 6-stage reactive state machine (FR-1, P1, P4, P6, P8)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, TypeVar, cast

from uclone_x.agent.grounding import describe, unsupported_specifics
from uclone_x.agent.hooks import (
    MD_IMAGE_RE,
    BaseHook,
    HookAction,
    HookContext,
    HookEvent,
    HookRunner,
    extract_artifact_rel_path,
    is_artifact_missing,
    sanitize_hallucinated_artifacts,
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
    TurnStopReason,
)
from uclone_x.agent.planner import ExecutionIntent, PlanGenerator
from uclone_x.agent.prompts import adapt_system_prompt
from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.agent.request_record import (
    RequestLayers,
    assemble_request_messages,
    compose_system_message,
    messages_digest,
    serialize_tools,
)
from uclone_x.agent.session import (
    AnchorAuthor,
    AnchorProvenance,
    CompactionResult,
    ContextSnapshot,
    SessionState,
    cleanup_session_artifacts,
    content_digest,
    redact_message,
    validate_session_id,
)
from uclone_x.agent.text_tool_calls import detect_text_emitted_tool_calls
from uclone_x.core.capability import Capability
from uclone_x.core.host import HostProtocol
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
    require_provenance,
)
from uclone_x.core.secrets import redact_credentials
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import (
    STEP_NO_ROOM_MESSAGE,
    STEP_OVER_WINDOW_MESSAGE,
    STEP_REPLY_RESERVE_TOKENS,
    TOOL_RESULT_READ_TOOL,
    artifacts_dir_for,
    canonical_tool_text,
    ingest_tool_text,
    step_result_caps,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventPriority,
    EventSource,
    EventType,
    UnauthorizedSubscriptionError,
)
from uclone_x.engine.protocols import (
    EventBusProtocol,
    EventSubscriptionProtocol,
    PublisherHandleProtocol,
)
from uclone_x.errors import (
    BudgetExceededError,
    InvalidStateTransitionError,
    LLMConnectorNotConfiguredError,
    LLMStreamInterruptedError,
    LLMTimeoutError,
    ModelLacksToolSupportError,
    PathTraversalError,
    SessionMutationDuringTurnError,
    SessionStoreNotConfiguredError,
    SessionSwitchWhileRunningError,
    TokenBudgetExhaustedError,
)
from uclone_x.llm.compactor import (
    ContextCompactor,
    estimate_message_tokens,
    estimate_reply_tokens,
    estimate_request_tokens,
    estimate_text_tokens,
    unseen_step_start,
)
from uclone_x.llm.context_window import (
    SERVED_WINDOW_PROVIDERS,
    OllamaContextWindows,
    compaction_window,
)
from uclone_x.llm.models import (
    BudgetDecision,
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenBudget,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
    aggregate_token_usages,
)
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
)
from uclone_x.llm.router import LLMTier, SemanticModelRouter
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
from uclone_x.telemetry.models import SpanStatus
from uclone_x.telemetry.protocols import TracerProtocol
from uclone_x.telemetry.tracer import FAILOVER_EVENT_SPAN_NAME, TelemetryTracer
from uclone_x.tools.base import (
    drop_shadowed_aliases,
    tool_spawns_subagents,
    tool_writes_files,
)
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.outcome import (
    EMPTY_RESULT_NOTE,
    ToolOutcome,
    classify_payload_shape,
    classify_tool_outcome,
)
from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_scoper import ToolScoperProtocol

logger = logging.getLogger(__name__)

# Upper bound on failures retained by `BaseAgent.processing_errors`.
_MAX_RECORDED_ERRORS = 100


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


#: How far up a `__cause__` chain `_is_provider_timeout` will look. A bound rather than a
#: `while`, because an exception chain can be made cyclic and a turn's error handler is
#: the last place that should be able to hang.
_MAX_CAUSE_DEPTH = 10

#: The tools whose paths resolve against the workspace; holding any one of them is what
#: makes the `[Workspace]` prompt section worth its tokens.
_FILE_TOOL_NAMES = frozenset(
    {"file_read", "file_write", "file_edit", "file_search", "directory_list"}
)


def _named_model(value: object) -> str | None:
    """`value` as a model name, or `None` when it names none (#1447).

    Empty, blank and the literal `"default"` name no model: `"default"` is the placeholder
    a request carries for "whatever the connector resolves", and recording it as the model
    that answered is the misattribution #1447 reports.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped and stripped != "default" else None


def _caller_turn_id_field(caller_turn_id: str | None) -> dict[str, str]:
    """`{"caller_turn_id": …}` when the caller gave one, else nothing to spread in."""
    return {} if caller_turn_id is None else {"caller_turn_id": caller_turn_id}


def _model_name_reason(model_name: str | None, why: str) -> dict[str, str]:
    """`{"model_name_reason": why}` when no model is named, else nothing to spread in."""
    return {} if model_name is not None else {"model_name_reason": why}


@dataclass(slots=True)
class _StreamProgress:
    """What one model call had produced when it stopped (#1489).

    `_invoke_model` fills it as chunks arrive, so the caller can record how far a call
    got when it raises -- including on `CancelledError`, which carries no payload and
    must be re-raised unchanged. `usage` is set only when the partial call was booked
    to the budget, so the record and the budget always carry the same figure.
    """

    content_chunks: list[str] = field(default_factory=lambda: [])
    thinking_chunks: list[str] = field(default_factory=lambda: [])
    usage: TokenUsage | None = None

    @property
    def content(self) -> str:
        return "".join(self.content_chunks)

    @property
    def thinking(self) -> str | None:
        return "".join(self.thinking_chunks) if self.thinking_chunks else None


_ChainedError = TypeVar("_ChainedError", bound=BaseException)


def _in_cause_chain(exc: BaseException, kind: type[_ChainedError]) -> _ChainedError | None:
    """The first exception of type `kind` in `exc`'s `__cause__` chain, `exc` included.

    The chain is walked rather than the exception tested directly, because the streaming
    path does not deliver the connector's error itself: `_invoke_model` raises
    `LLMStreamInterruptedError` with the provider's error chained as `__cause__`. Testing
    only the outermost type would report a cut-off stream as an ordinary failure.
    """
    seen = exc
    for _ in range(_MAX_CAUSE_DEPTH):
        if isinstance(seen, kind):
            return seen
        cause = seen.__cause__
        if cause is None:
            return None
        seen = cause
    return None


def _is_provider_timeout(exc: BaseException) -> bool:
    """Whether a turn failed because a provider ran past the ceiling the caller set."""
    return _in_cause_chain(exc, LLMTimeoutError) is not None


def _model_lacking_tools(exc: BaseException) -> ModelLacksToolSupportError | None:
    """The refusal of a model that cannot take tools, if that is why a turn failed.

    The error the turn reports when it is found: its message is written for the user,
    while the `LLMStreamInterruptedError` wrapping it names classes and a status line.
    """
    return _in_cause_chain(exc, ModelLacksToolSupportError)


def _turn_failure(
    exc: BaseException, stop_reason: TurnStopReason, agent_id: str
) -> tuple[TurnStopReason, str]:
    """The stop reason and `error` a turn that raised `exc` ends with, logged as it deserves.

    A function of its own because `execute_turn` is at the edge of what the type checker
    can analyse, and each branch added inline pushes it over.
    """
    if _is_provider_timeout(exc):
        stop_reason = "provider_timeout"
    lacking_tools = _model_lacking_tools(exc)
    if lacking_tools is None:
        logger.exception("Error executing turn for agent %s", agent_id)
        return stop_reason, str(exc)
    # A choice of model, not a fault in the code. With no handler set up, Python prints
    # WARNING and above -- traceback included -- on the user's terminal, beside the plain
    # sentence the CLI already shows.
    logger.info("Turn for agent %s refused: %s", agent_id, lacking_tools)
    return "model_without_tools", str(lacking_tools)


class _AnchorWriter(Enum):
    """The anchor provenances that are not a persona resolution.

    Two members, because there are exactly two things a persona resolution cannot
    express: "this agent did not compose the anchored system turn", and "nothing on
    record says what composed it". They are **not** the same claim and collapsing them
    would be a silent fallback (P6): the first is knowledge, the second is its absence,
    and only the second is a condition worth reporting. Both are treated alike by
    `_anchor_is_stale` — an anchor with no position behind it has nothing to compare —
    and that shared answer is a consequence, not the reason they are one type.

    An `Enum` rather than bare `object()`s so the type checker can see them in a union.
    """

    CALLER = "caller"
    UNRECORDED = "unrecorded"


#: What a session's anchored system turn was composed from. A `PersonaDefinition` — or
#: `None` for "the axis resolved to no persona" — is *this agent's own* resolution at the
#: moment it wrote that anchor, which `_anchor_is_stale` compares against the axis as it
#: stands when a turn is built. `_AnchorWriter.CALLER` is text that arrived from outside
#: (`load_history`): the agent did not compose it and has no axis position to attribute it
#: to, so it is not re-resolved. `_AnchorWriter.UNRECORDED` is a session restored from a
#: record that carries no `anchor_provenance` — written before the field existed, or built
#: by a door with no answer to give (#1152).
_AnchorProvenance = PersonaDefinition | None | _AnchorWriter


def _has_anchor(messages: Sequence[ChatMessage]) -> bool:
    """Whether `messages` opens with a `SYSTEM` turn — the anchor the turn builder re-frames.

    One expression for the two places that ask (#1174, #1152): the turn builder, which
    sends what `effective_system_prompt` reports when there is no anchor, and
    `hydrate_session`, which only reports missing provenance for a record that actually
    has an anchor for the provenance to be missing *about*. A record of user-first rows
    has nothing to re-resolve, so warning about it would be noise.
    """
    return bool(messages) and messages[0].role == MessageRole.SYSTEM


def compose_identity_prompt(
    *,
    config_prompt: str,
    persona: PersonaDefinition | None,
    seat_framing: str = "",
) -> str:
    """The identity layer: the one function that builds the prompt an agent speaks as.

    The anchor a session stores and the system message a turn sends both come from here,
    through `BaseAgent._system_prompt_base`. The room used to assemble a seat's prompt
    itself and hand the result in after the agent had already seeded its session. The
    stored anchor then lacked the seat's framing while the turn sent it, so the record
    and the wire disagreed.

    * No framing: the persona's prompt when one is in force, else `config_prompt`, as
      before.
    * Framing and a persona with instructions: the framing, then the persona's
      instructions under a labelled header.
    * Framing and nothing else to say: the framing, then `config_prompt`.
    """
    body = persona.system_prompt if persona is not None else config_prompt
    if not seat_framing:
        return body
    if persona is not None and persona.system_prompt:
        return f"{seat_framing}\n\n[Persona Instructions: {persona.role}]\n{persona.system_prompt}"
    return f"{seat_framing}\n\n{config_prompt}"


#: Opens the block of state that changes during a conversation. It travels at the tail of
#: the request, never in the system turn: see `_turn_context_block`.
TURN_CONTEXT_HEADER = (
    "[Turn Context]\n"
    "Supplied by the runtime with every request, not written by the user. It states the "
    "current value of state that changes during the conversation; where it disagrees "
    "with an earlier turn, this is the current one."
)


def _turn_context_block(sections: Sequence[str]) -> str:
    """The `[Turn Context]` block for `sections`, placed at the tail of the request.

    Empty when there are none. `place_turn_context` puts it at the tail, never in the
    system turn.

    **The system turn is the head of every prefix a provider or a local server can reuse.**
    One changed byte there re-prefills the whole conversation behind it -- on a local
    model the latency the conversation waits on, on a hosted one the full input price.
    The plan (a box ticked per completed step), cross-session memory (a fact recorded
    mid-conversation, inserted by confidence rather than appended) and the tool-scoping
    notice (recomputed from each prompt) all change inside a conversation, so while they
    lived in the system turn every such change discarded the entire cached prefix.

    At the tail a change costs only the block itself. The block is built per request and
    never written to history, so the next request's prefix -- system turn plus history --
    is byte-identical to this one's up to where this block began.

    A `USER` role, not `SYSTEM`: the Anthropic and Gemini connectors hoist every `SYSTEM`
    message, wherever it stands, into the one top-level system field, which would put the
    block straight back at the head. When the request already ends with the user's
    message (a turn's first step) the block joins that message rather than following it
    as a second consecutive user turn, which chat templates that require alternating
    roles refuse; after a tool result there is no user message to join, and it follows.
    """
    if not sections:
        return ""
    return "\n\n".join((TURN_CONTEXT_HEADER, *sections))


def _persisted_anchor_provenance(provenance: _AnchorProvenance) -> AnchorProvenance | None:
    """Render a live stamp into the shape `SessionState` persists (#1152).

    `UNRECORDED` renders as `None` — "the record still does not say" — rather than as
    anything the reader could mistake for a stamp. A session restored from a legacy
    record and persisted again is honest about the gap instead of inventing a filling for
    it, and the *next* anchor this agent composes stamps itself properly.
    """
    if provenance is _AnchorWriter.UNRECORDED:
        return None
    if provenance is _AnchorWriter.CALLER:
        return AnchorProvenance(author=AnchorAuthor.CALLER)
    return AnchorProvenance(author=AnchorAuthor.AGENT, persona=provenance)


def _restored_anchor_provenance(record: AnchorProvenance | None) -> _AnchorProvenance:
    """Read a persisted stamp back into the live form, or report that there is none.

    The inverse of `_persisted_anchor_provenance`, and the whole of what #1152 asked for:
    with the stamp on the record, a restored session can say which persona composed its
    anchor instead of every restored anchor reading as the caller's and never being
    re-resolved.

    A record with no stamp reads as `UNRECORDED`, never as `None`. `None` here means "the
    agent composed this under no persona", which is a claim — and a false one would make
    the very next persona adoption overwrite an anchor the caller may have supplied.
    """
    if record is None:
        return _AnchorWriter.UNRECORDED
    if record.author is AnchorAuthor.CALLER:
        return _AnchorWriter.CALLER
    return record.persona


@dataclass(slots=True)
class _LiveSession:
    """Mutable working copy of one session's conversation.

    The agent mutates a session on the hot path — `_history.append` on every turn — so
    the live form is a list, while `SessionState` is the frozen shape the store persists.

    **Exactly one object per session id holds the messages on this agent instance.**
    That is the per-agent invariant, and it is what the abandoned `3a8b7d6` attempt at this
    issue got wrong in two places: it kept `self._history` as an *alias* into
    `self._sessions[sid]`, so any rebinding of the dict entry silently detached the two
    names for one piece of state; and its per-session `_turn_counters` were written only by
    `__init__`, `load_history` and `reset_session`, never by a turn, so `switch_session`
    away and back reported a turn count of zero for a session that had run turns. Here
    `BaseAgent._history` and `BaseAgent._turn_counter` are properties onto this object, so
    there is no second location to fall out of sync.
    """

    messages: list[ChatMessage]
    plan: PlanState | None
    turn_counter: int
    created_at: str
    updated_at: str
    #: The store revision this working copy was last in agreement with — the revision it
    #: hydrated at, or the one its last successful `save` wrote. It is carried here rather
    #: than recomputed because it is the only thing that makes the store's compare-and-swap
    #: reach across a turn: without it every `to_state` would claim revision 0 and either
    #: be refused forever or, worse, silently pass on a record that happened to still be
    #: at 0. `0` means this session has never round-tripped through a store.
    revision: int = 0
    #: What this session's anchored system turn was composed from, stamped by whichever
    #: call wrote that anchor (#1081). It lives here, per session, because that is what it
    #: describes: the same agent can hold one session seeded under a persona and another
    #: hydrated from a caller's own text, and an agent-wide flag answers for both at once.
    #: It survives the store as `SessionState.anchor_provenance` (#1152): `to_state` renders
    #: it through `_persisted_anchor_provenance` and `hydrate_session` reads it back through
    #: `_restored_anchor_provenance`, so a restored session says which persona composed its
    #: anchor instead of attributing every restored anchor to the caller. A record that
    #: predates the field comes back `UNRECORDED` — reported, and still not re-resolved.
    anchor_provenance: _AnchorProvenance = _AnchorWriter.CALLER
    #: What each turn's requests carried besides the conversation (#1421). Persisted as
    #: `SessionState.context_snapshots`; the last one is reused while nothing in it changes.
    context_snapshots: list[ContextSnapshot] = field(default_factory=list[ContextSnapshot])
    #: The conversation the previous request of this session sent, as recorded, so the
    #: next `REQUEST_CONTEXT` records only what was added. Not persisted: after a restart
    #: the first request records its whole conversation once and the chain starts again.
    last_conversation: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    #: The number of the previous request in this chain, or `None` before the first. Each
    #: event names the request it extends, so a reader can tell a gap in the log apart
    #: from a request that really did extend the one before it.
    last_request: int | None = None
    #: Body digests already written to the store by this working copy.
    stored_bodies: set[str] = field(default_factory=set[str])
    #: Bodies the snapshots name that are not written yet, by digest. Every save
    #: writes them before the record that names them, so a failed write fails the save
    #: that reports it rather than the turn: the reply is real either way.
    pending_bodies: dict[str, str] = field(default_factory=dict[str, str])
    #: The tool calls the last turn asked for, as its `TOOL_CALL` events record them.
    #: Cleared by `checkpoint_turn`, so a rollback names only calls made after its
    #: checkpoint (#1495).
    last_turn_tool_calls: list[ToolCallRequest] = field(default_factory=list[ToolCallRequest])
    #: Calls made by attempts that were rolled back and not yet followed by a turn that
    #: stayed, stated to the next turn in its turn context (#1495). Not persisted: see
    #: the request-layering design, §5.6.
    undone_tool_calls: list[ToolCallRequest] = field(default_factory=list[ToolCallRequest])
    #: Whether a turn has shown `undone_tool_calls` since the last rollback. The next turn
    #: to start finds it set only if that turn was not rolled back, and clears the list.
    undone_tool_calls_shown: bool = False

    @classmethod
    def from_state(
        cls, state: SessionState, *, anchor_provenance: _AnchorProvenance
    ) -> _LiveSession:
        """Adopt a persisted or seeded session as the live working copy.

        `anchor_provenance` is keyword-only and has no default here, so a call that omits
        it does not type-check: deciding who composed the anchor is part of putting a
        session into `_sessions`, and a default would let a new call inherit that answer
        by omission. The field's own default exists for the dataclass, not for callers.
        """
        return cls(
            messages=list(state.messages),
            plan=state.plan,
            turn_counter=state.turn_counter,
            created_at=state.created_at,
            updated_at=state.updated_at,
            revision=state.revision,
            anchor_provenance=anchor_provenance,
            context_snapshots=list(state.context_snapshots),
        )

    def to_state(self, session_id: str, agent_id: str) -> SessionState:
        """Snapshot this session into the frozen shape the store persists.

        **This is the only door that writes `anchor_provenance` onto a record**, because
        it is the only place the answer is held. Every route to `SessionStore.save` that
        an agent takes passes through here, so stamping it here rather than at each save
        is what keeps the record's account of its anchor and the anchor itself together.
        """
        return SessionState(
            session_id=session_id,
            agent_id=agent_id,
            messages=tuple(self.messages),
            plan=self.plan,
            turn_counter=self.turn_counter,
            created_at=self.created_at,
            updated_at=self.updated_at,
            revision=self.revision,
            anchor_provenance=_persisted_anchor_provenance(self.anchor_provenance),
            context_snapshots=tuple(self.context_snapshots),
        )


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


#: Sent back to a model that answered with no tool execution, when the agent is configured
#: to require evidence. Phrased as a question about grounding rather than an instruction to
#: use a named tool: naming one steers the choice, and an earlier measurement of exactly
#: that steer moved one problem of ten -- inside the noise for an unrepeated run (#698).
EVIDENCE_REQUIRED_NUDGE = (
    "Nothing you did this turn produced evidence: either no tool ran, or the ones that ran "
    "returned nothing. A tool that matched nothing has not shown the thing is absent -- the "
    "query or the tool may be the wrong one. If the answer rests on something you can check "
    "here, check it a different way and answer again. If this question is answerable from what "
    "you were given, say that it is answerable from what you were given. If it does not, say "
    "plainly that you could not verify it."
)


DECLINE_PHRASES: tuple[str, ...] = (
    "answerable from what you were given",
    "answerable from what was given",
    "answerable from the prompt",
    "answerable from the given",
    "answerable from given",
    "answerable without",
    "derivable from the given",
    "derivable from given",
    "deducible from the given",
    "deducible from given",
    "without requiring external tools",
    "without tool use",
    "no tool calls were needed",
    "no tools were needed",
    "self-contained",
)


def _extract_terminal_answer(text: str) -> str | None:
    match = re.search(
        r"(?im)(?:final\s+answer|answer|\*\*answer\*\*):\s*[\*]*([a-zA-Z0-9_-]+)[\*]*",
        text,
    )
    if match:
        return match.group(1).lower()
    first_match = re.match(r"(?im)^\s*(?:answer:\s*)?[\*]*([a-zA-Z0-9_-]+)[\*]*[\.\!:,]", text)
    if first_match:
        val = first_match.group(1).lower()
        if val in {"yes", "no", "trapped", "correct"}:
            return val
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = re.sub(r"[\*_\.<>]", "", lines[-1]).strip().lower()
        if len(last_line.split()) <= 2:
            return last_line
    return None


def _is_evidence_nudge_declined(content: str, first_answer: str | None = None) -> bool:
    """Whether the model declined the evidence nudge.

    Occurs either explicitly (declaring the prompt self-contained or answerable without tools)
    or implicitly when the post-nudge response is materially equivalent to the first answer
    without tool use (#784).
    """
    lowered = content.lower()
    if any(phrase in lowered for phrase in DECLINE_PHRASES):
        return True

    if first_answer is None:
        return False

    lowered_first = first_answer.lower().strip()
    lowered_second = lowered.strip()

    # Conflicting answers (e.g. opposite boolean polarity or contradictory numbers) do not decline
    nums1 = set(re.findall(r"\b\d+\b", lowered_first))
    nums2 = set(re.findall(r"\b\d+\b", lowered_second))
    if nums1 and nums2 and nums1.isdisjoint(nums2):
        return False

    has_yes1 = bool(re.search(r"\byes\b", lowered_first))
    has_no1 = bool(re.search(r"\bno\b", lowered_first))
    has_yes2 = bool(re.search(r"\byes\b", lowered_second))
    has_no2 = bool(re.search(r"\bno\b", lowered_second))
    if (has_yes1 and not has_no1 and has_no2 and not has_yes2) or (
        has_no1 and not has_yes1 and has_yes2 and not has_no2
    ):
        return False

    # If second is an abstention/retreat and first was not, the model complied with
    # the nudge's request to abstain rather than declining it.
    abstention_patterns = (
        r"\bcould not verify\b",
        r"\bcannot verify\b",
        r"\bunable to verify\b",
        r"\bnot verified\b",
    )
    is_second_abstention = any(re.search(pat, lowered_second) for pat in abstention_patterns)
    is_first_abstention = any(re.search(pat, lowered_first) for pat in abstention_patterns)
    if is_second_abstention and not is_first_abstention:
        return False

    # Exact string match (ignoring leading/trailing whitespace)
    if lowered_first == lowered_second:
        return True

    # Terminal answer matches (e.g. "Answer: no" or "5")
    term_first = _extract_terminal_answer(first_answer)
    term_second = _extract_terminal_answer(content)
    if term_first and term_second and term_first == term_second:
        return True

    # Concise first answer appears as a distinct phrase in post-nudge answer
    if len(lowered_first) <= 60:
        clean_first = re.sub(r"[\*_\.<>]", "", lowered_first).strip()
        pattern = rf"\b{re.escape(clean_first)}\b"
        # An empty phrase makes the pattern `\b\b`, which matches any answer at all.
        if clean_first and re.search(pattern, lowered_second):
            return True

    # High word-level Jaccard similarity (>= 0.6)
    w1 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", lowered_first))
    w2 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", lowered_second))
    if w1 and w2 and (len(w1 & w2) / len(w1 | w2) >= 0.6):
        return True

    return False


#: Opens the nudge sent back when the answer names specifics the turn read nowhere. The
#: constant is the stable prefix; the findings are appended, so a reader can recognise the
#: message without predicting which tokens it will quote.
#:
#: Two exits, deliberately. "Check it" alone is a refusal wearing a question -- it blocks a
#: task whose answer was legitimately derived rather than read, and a derived figure (a
#: count, a difference) is correct and appears in no output. "Say it is unchecked" is the
#: cheaper of the two errors under P6 and is what `DEFAULT_SYSTEM_PROMPT` already asks for.
#: As with the evidence nudge, no tool is named: naming one steers the choice (#698).
GROUNDING_REQUIRED_NUDGE_PREFIX = (
    "Your answer states specifics that appear in nothing you have read this turn: "
)

GROUNDING_REQUIRED_NUDGE_SUFFIX = (
    ". Check them here and answer again, or keep the answer and say plainly which parts of "
    "it you did not verify."
)


def _grounding_supports(
    messages: Sequence[ChatMessage],
    tool_executions: Sequence[ToolExecutionRecord],
    injected: Collection[str] = (),
) -> list[str]:
    """Everything the turn was shown or told, minus what it said itself.

    Two exclusions, and both were measured rather than reasoned about:

    *   **Assistant turns.** A claim cannot be its own evidence. With them included, the
        answer given after a nudge finds its figure in the answer that provoked the nudge
        and reads as grounded.
    *   **`injected`, the runtime's own nudges.** The grounding nudge *quotes the
        unsupported specifics back*, so it lands in the request carrying exactly the tokens
        that are missing. Left in the support set it grounds them, and the second look at a
        repeated answer comes back clean -- the check silently disarming itself one step
        after firing. Pinned by `test_the_nudge_quoting_a_specific_does_not_then_ground_it`;
        the assistant exclusion is pinned by
        `test_the_models_own_earlier_answer_does_not_support_its_later_one`.

    A nudge is cut out of the message that carries it rather than matched against the
    whole message: it travels inside the turn-context block (#1420), so the message holds
    the nudge and more, and an equality test would never exclude it.
    """
    supports: list[str] = []
    for m in messages:
        if m.role is MessageRole.ASSISTANT or not m.content:
            continue
        text = str(m.content)
        for nudge in injected:
            text = text.replace(nudge, "")
        supports.append(text)
    supports.extend(str(record.output) for record in tool_executions)
    return supports


def _tool_outcome_of(record: ToolExecutionRecord) -> str:
    """The errored / empty / productive classification, for the log.

    `ToolExecutionRecord` is what the turn keeps; `classify_tool_outcome` reads a
    `ToolResult`. Rather than thread the result through, the record's own fields are
    mapped the same way -- one place, and the mapping is asserted against the classifier
    in `tests/unit/test_turn_step_logging.py` so the two cannot drift.
    """
    if record.status != ToolResultStatus.SUCCESS:
        return ToolOutcome.ERRORED.value
    return classify_payload_shape(record.output).value


def _request_context_delta(
    previous: Sequence[dict[str, Any]], current: Sequence[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """What a request's conversation adds to the previous one's: `(kept, appended)` (#1442).

    `kept` is the length of the prefix `current` shares with `previous`, and `appended` is
    the rest of `current`. The durable `REQUEST_CONTEXT` event records these instead of the
    whole conversation, because the whole of it was re-recorded on every step: an N-step
    turn wrote N copies of a history that grows each step, so the event log grew
    quadratically in N. A request's conversation is the previous one plus what the step
    added, so `appended` is that delta and the log grows linearly. The prefix is compared,
    not assumed, so a conversation that rewrites an earlier message -- compaction, a
    history edit -- records everything from the first difference, and the full
    conversation is always `previous[:kept] + appended`.
    """
    kept = 0
    for before, now in zip(previous, current, strict=False):
        if before != now:
            break
        kept += 1
    return kept, list(current[kept:])


def _repeats_unanswered_prompt(history: Sequence[ChatMessage], user_prompt: ChatMessage) -> bool:
    """Whether `user_prompt` is the prompt already waiting, unanswered, at the end of `history`.

    A turn puts its prompt in history before it calls the model, and a turn that fails
    leaves it there (#969). A retry sends the same prompt again -- the chat page's Retry
    sends it as a new turn, and a room's retry renders the same unseen span -- so appending
    it a second time showed the model two copies of one question. Nothing answered the
    first copy, so a second one adds nothing a model could use. A prompt that *was*
    answered is followed by the reply, so the same words sent again are a new message and
    are kept.

    **The exception is an answer that was empty.** The step loop appends nothing when a
    turn produced neither content nor a tool call (`if tool_calls or resp_content:`), so
    history still ends with the prompt and an identical re-send is read as a repeat even
    though the turn ran and returned. The re-send still runs and is still answered; what
    it does not do is put the words in front of the model twice.

    `name` is compared beside the text because it is what distinguishes one sender from
    another in a transcript that names them (`reconstruct_history` sets it from a stored
    prompt). Only a sender's own unanswered prompt is a repeat of it: were a head to name
    its users, two of them sending the same words would otherwise collapse into one.
    """
    if not history:
        return False
    last = history[-1]
    return (
        last.role is MessageRole.USER
        and last.content == user_prompt.content
        and last.name == user_prompt.name
    )


def _unanswered_tool_step(history: Sequence[ChatMessage]) -> tuple[int, tuple[str, ...]] | None:
    """Where the last step's tool calls went unanswered: `(index, call_ids)`, or `None` (#1423).

    A step appends the `ASSISTANT` message that asked for tools *before* it awaits them,
    and appends their `TOOL` results only once all of them return. A turn stopped between
    the two -- Stop, an interjection, a failure inside the tool round -- leaves a message
    that asks for calls nothing answers. Every provider refuses such a conversation, so
    the session could not be sent again, and the next turn was the one to find out.

    Only the last `ASSISTANT` message is examined: every earlier step's round completed
    before the next model call, or the turn could not have reached a later step.
    """
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.role is not MessageRole.ASSISTANT:
            continue
        asked = tuple(call.id for call in message.tool_calls)
        if not asked:
            return None
        answered = {
            later.tool_call_id for later in history[index + 1 :] if later.role is MessageRole.TOOL
        }
        missing = tuple(call_id for call_id in asked if call_id not in answered)
        return (index, missing) if missing else None
    return None


#: Longest rendering of one argument value, and of one call's whole argument list, in the
#: undone-attempt statement. A summary says which call it was; the call itself is in the log.
_UNDONE_ARG_VALUE_CHARS: Final = 40
_UNDONE_ARGS_CHARS: Final = 120
#: Calls listed before the rest are counted rather than named.
_UNDONE_CALLS_LISTED: Final = 20


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_call(call: ToolCallRequest) -> str:
    """`name(key=value, ...)`, each value and the whole list clipped; never a result."""
    parts: list[str] = []
    for key, value in call.arguments.items():
        rendered = json.dumps(unwrap_immutable(value), ensure_ascii=False, default=str)
        parts.append(f"{key}={_clip(rendered, _UNDONE_ARG_VALUE_CHARS)}")
    return redact_credentials(f"{call.name}({_clip(', '.join(parts), _UNDONE_ARGS_CHARS)})")


def _undone_attempt_section(calls: Sequence[ToolCallRequest]) -> str:
    """The turn-context statement of what an undone attempt called, or `""` (#1495).

    A rolled-back attempt leaves the conversation, but not the world: a tool it ran may
    have saved a memory or written a file. Without this, the retry is shown the same
    history the attempt started from and may run a non-idempotent call a second time.
    A step refused for the context window is withheld from history after its tools ran,
    so its calls are stated here too (#1509).

    **A runtime statement in the turn context, not a `TOOL` message.** A `TOOL` message
    is a tool's own result; one the runtime composed would be the substituted result P6
    forbids, and it would enter history permanently. The statement is layer 5 of the
    request-layering design (§5): built per request, never written to history, recorded
    verbatim in the turn's context snapshot. It names each call and a clipped summary of
    its arguments, never an output, and says only what is known: the call was made, and
    undoing the attempt did not undo it.
    """
    if not calls:
        return ""
    listed = [f"- {_summarize_call(call)}" for call in calls[:_UNDONE_CALLS_LISTED]]
    if len(calls) > _UNDONE_CALLS_LISTED:
        listed.append(f"- and {len(calls) - _UNDONE_CALLS_LISTED} more")
    return (
        "[Undone Attempt]\n"
        "An earlier attempt was stopped and undone, so its messages are not in this "
        "conversation. It called the tools below. Undoing the attempt did not undo what "
        "they did, and a call may have run even where no result was recorded.\n" + "\n".join(listed)
    )


def _detect_missing_artifacts(
    resp_content: str,
    tools: ToolRegistryProtocol | None,
    workspace_root: Path | None,
) -> list[str]:
    """Find referenced image artifacts in response content that do not exist on disk."""
    if tools is None or tools.get("generate_image") is None:
        return []
    missing: list[str] = []
    for match in MD_IMAGE_RE.finditer(resp_content):
        raw_url = match.group(2)
        rel = extract_artifact_rel_path(raw_url)
        if rel and (rel.startswith("artifacts/images/") or rel.startswith("artifacts/")):
            if is_artifact_missing(rel, workspace_root):
                missing.append(rel)
    return missing


def _evaluate_artifact_nudge(
    resp_content: str | None,
    tools: ToolRegistryProtocol | None,
    workspace_root: Path | None,
    artifact_nudged: bool,
) -> tuple[str, str] | None:
    """Check if in-turn nudge is needed for missing artifact image links."""
    if artifact_nudged or not resp_content:
        return None
    missing = _detect_missing_artifacts(resp_content, tools, workspace_root)
    if not missing:
        return None
    first_missing = Path(missing[0]).name
    nudge = (
        f"The referenced image file '{first_missing}' does not exist on disk, and no image generation "
        f"tool was called to produce it. Never predict, invent, or guess image URLs. "
        f"To provide an image, you must call the 'generate_image' tool with your prompt."
    )
    return missing[0], nudge


def _apply_artifact_sanitization(
    content: str,
    workspace_root: Path | None,
    history: list[ChatMessage],
    assistant_msg_idx: int | None,
) -> str:
    """Post-turn sanitization for any remaining hallucinated artifact images."""
    if not content:
        return ""
    sanitized, count = sanitize_hallucinated_artifacts(content, workspace_root)
    if count > 0:
        if assistant_msg_idx is not None and assistant_msg_idx < len(history):
            orig_msg = history[assistant_msg_idx]
            history[assistant_msg_idx] = redact_message(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=sanitized or None,
                    tool_calls=orig_msg.tool_calls,
                )
            )
        return sanitized
    return content


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
        self._persona_store[persona.name] = persona
        self._apply_persona_tool_scope()  # a registration can move the persona in force

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Resolve a persona: this agent's own registrations first, then the registry.

        There is no third branch. `scout` and `critic` were returned from module constants
        ahead of the store, so a persona registered under either name was silently ignored
        rather than overriding or being refused; both now ship as YAML that the registry
        seeds, so the constants have no resolution role left.
        """
        if name in self._persona_store:
            return self._persona_store[name]
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
        tool_scoper: ToolScoperProtocol | None = None,
        plan_generator: PlanGenerator | None = None,
        persona: str | None = None,
        persona_name: str | None = None,
        skills: SkillRegistryProtocol | None = None,
        host: HostProtocol | None = None,
        memory: CrossSessionMemory | None = None,
        personas: Sequence[PersonaDefinition] = (),
    ) -> None:
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
        self._bus = bus
        self._llm = llm
        self._tools = tools
        self._skills = skills
        self._loaded_skills: set[str] = set()
        # Tools bound to *this* agent's own state, consulted by `_resolve_tool` before the
        # registry. The UI gives all four heads one `ToolRegistry`, so a tool registered
        # there and resolved from there is whichever agent's instance was composed first.
        # The registry copy still exists for *advertisement*: the schema and description
        # are agent-independent, and registering there is what keeps a persona's
        # `allowed_tools` whitelist able to scope them out (#1097).
        self._agent_local_tools: dict[str, ToolProtocol] = {}
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
            skill_tool = LoadSkillTool(self._skills, on_load=self._record_loaded_skill)
            self._agent_local_tools[skill_tool.name] = skill_tool
            if self._tools.get(skill_tool.name) is None:
                self._tools.register(skill_tool)
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
                self._agent_local_tools[memory_tool.name] = memory_tool
                if self._tools.get(memory_tool.name) is None:
                    self._tools.register(memory_tool)
        self._store = store
        self._injected_compactor = compactor
        self._budget = budget
        # Agent steps taken in the current run — cleared by every externally initiated
        # turn. Distinct from `_turn_counter`, the lifetime count the session persists.
        self._run_steps = 0
        self._steps_deducted = False
        self._pending_durable_events: list[dict[str, Any]] = []
        self._semantic_router = semantic_router
        self._tool_scoper = tool_scoper
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
        if isinstance(session.anchor_provenance, _AnchorWriter):
            return False
        return session.anchor_provenance != self._resolved_persona()

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
        resolved = self._operator_allowed_tools
        persona = self._resolved_persona()
        if not resolved and persona is not None and persona.granted_tools:
            resolved = persona.granted_tools
        if resolved != self._config.allowed_tools:
            self._config = self._config.model_copy(update={"allowed_tools": resolved})

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
    ) -> None:
        """Hot-reload active LLM connector and optional model or system prompt configuration without restart (Issue #350, #871)."""
        if llm is not None:
            self._llm = llm
        updates: dict[str, Any] = {}
        if model_name is not None:
            new_llm_config = self._config.llm_config.model_copy(update={"model_name": model_name})
            updates["llm_config"] = new_llm_config
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
        self._active_session.messages = list(messages)
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

    def _effective_session_id(self, session_id: str | None) -> str:
        """Resolve an optional session argument to a concrete id.

        `None` means "the active session". An empty string does **not**: the `or` idiom
        this replaces treated `""` as absent, so `persist_session("")` silently wrote the
        *active* session under the active id, and `reset_session("")` reset a session the
        caller had not named. That is the empty-id ambiguity `validate_session_id`
        refuses one layer down, and it should not be reintroduced by an idiom here.
        """
        return self._context.session_id if session_id is None else session_id

    def _refuse_session_mutation_during_turn(self, session_id: str, operation: str) -> None:
        """Refuse a reset or switch that would corrupt a turn in flight **on this agent**.

        `execute_turn` holds `_turn_lock` for the whole turn and appends to the active
        session as it goes, so mutating that session underneath it produces a transcript
        with an answer and no question. Only the active session is at risk: a turn never
        touches another one, so a mutation aimed elsewhere is allowed.

        **Scope, stated because the obvious reading is wider than the truth.**
        `_turn_lock` is an `asyncio.Lock` on *this instance*, so this guard sees only
        this agent's turns. **Since #225 it is also the only guard on a switch.** A
        running agent's `switch_session` used to raise `SessionSwitchWhileRunningError`
        unconditionally, which incidentally made every mid-turn switch on a started agent
        unreachable; that cover is gone now that switching is legal, and this check is
        what remains. Two `BaseAgent` objects addressing the same session id — the
        UI can hold one per `(agent_id, session_id)` key while the CLI holds another over
        the same `SessionStore` — are outside its reach by construction: agent `q` may
        reset a session while agent `p` is mid-turn on it, and memory and disk diverge
        with nothing to observe it. That is cross-instance concurrency on one record,
        which no per-instance lock can arbitrate; it needs optimistic concurrency on the
        store, which is carried separately (#219). This guard claims only the
        per-instance property, and holds it.
        """
        if not self._turn_lock.locked():
            return
        if session_id != self._context.session_id:
            return
        raise SessionMutationDuringTurnError(
            f"Agent '{self.agent_id}' cannot {operation} session '{session_id}' while a "
            "reasoning turn is in flight on it: the turn would complete into the mutated "
            "session and report an answer to a question that is no longer in its history. "
            "Await the turn first."
        )

    def _seed_live_session(self, session_id: str) -> _LiveSession:
        """A freshly seeded live session, stamped with the axis position it was seeded at.

        `__init__` and `_live_session` both seed, and they seeded through two copies of the
        same `SessionState.seed` call. One copy here is what keeps the seeded prompt and
        the provenance stamped beside it from drifting apart: a stamp that named a
        different persona than the anchor it describes would be worse than no stamp, since
        `_anchor_is_stale` would then answer confidently and wrongly.
        """
        return _LiveSession.from_state(
            SessionState.seed(
                session_id=session_id,
                agent_id=self._config.agent_id,
                system_prompt=self.effective_system_prompt,
            ),
            anchor_provenance=self._resolved_persona(),
        )

    def _live_session(self, session_id: str) -> _LiveSession:
        """Return the live session for `session_id`, seeding it if it is new.

        Seeding on first touch is what makes a session id sufficient to address a
        conversation: a caller does not have to create a session before using it, and a
        newly addressed session starts from the same seeded state as the agent's first.

        The id is validated first, by the same rule the store applies. An agent that
        accepted `""` or `"../../../etc/evil"` as a session name would host a
        conversation that can never be persisted — the store refuses it at write time —
        and the failure would surface far from the call that caused it.
        """
        validate_session_id(session_id)
        live = self._sessions.get(session_id)
        if live is None:
            live = self._seed_live_session(session_id)
            self._sessions[session_id] = live
        return live

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
        """Snapshot one hosted session as frozen `SessionState`.

        Reading an unknown session id returns its synthesized seeded state without
        mutating `_sessions`, ensuring reading arbitrary session IDs is a pure, idempotent
        operation that does not cause unbounded in-memory cache growth.
        """
        if session_id is None:
            active_id = self._context.session_id
            return self._live_session(active_id).to_state(active_id, self._config.agent_id)

        validate_session_id(session_id)
        live = self._sessions.get(session_id)
        if live is not None:
            return live.to_state(session_id, self._config.agent_id)

        return SessionState.seed(
            session_id=session_id,
            agent_id=self._config.agent_id,
            system_prompt=self.effective_system_prompt,
        )

    def _subscription_topics(self, session_id: str) -> set[str]:
        """The topic set this agent listens on while active in `session_id`.

        `start` and `switch_session` both call it so the topics a running agent holds
        cannot drift from the topics it was started with — the drift being exactly the
        mis-routing #225 describes, one refactor away.
        """
        return {
            f"agent.{self.agent_id}",
            f"session.{session_id}",
            "broadcast",
        }

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
        """Make `session_id` the active session, seeding it if new.

        Legal while the agent is running (#225). The bus subscription is **repointed in
        place** rather than closed and re-opened: `EventSubscription.retarget` swaps the
        topic set and session filter on the live object, so the queue survives and the
        `agent.{id}` and `broadcast` events already buffered on it are re-queued instead
        of dying with a closed subscription.

        **What a switch does discard, since it is not nothing.** Events already queued
        for the session being left no longer match the new target. They cannot be
        answered after the switch — `execute_turn` runs against whatever is active, so
        delivering them would file the reply in the wrong conversation, which is the
        defect this method used to refuse outright. They are therefore dropped, and
        `stranded_event_counts` reports how many, per session. Nothing is discarded
        silently.

        Raises:
            SessionMutationDuringTurnError: If a reasoning turn is in flight on the
                session being left. The turn appends to whatever is active, so switching
                underneath it lands the assistant message in the wrong conversation.
                Checked first, and independently of the bus: a not-yet-started agent, or
                one built with `bus=None`, has no subscription, and this guard is now the
                *only* thing standing between a concurrent caller and a corrupted
                transcript — `SessionSwitchWhileRunningError` no longer blocks the
                running case, so its incidental cover is gone.
            PathTraversalError: If `session_id` is not a legal session name. Validated
                before the subscription is touched.
            SessionSwitchWhileRunningError: If the bus refuses the new session's topic —
                its allowlist or wildcard-capability rules reject `session.{session_id}`.
                The agent would then report a session it cannot receive events for, so
                the switch fails whole: the subscription, the active session id and the
                hosted-session map are all left as they were.
        """
        self._refuse_session_mutation_during_turn(self._context.session_id, "switch away from")
        validate_session_id(session_id)

        previous_id = self._context.session_id
        if self._running and self._subscription is not None and session_id != previous_id:
            try:
                stranded = self._subscription.retarget(
                    self._subscription_topics(session_id),
                    session_id=session_id,
                )
            except UnauthorizedSubscriptionError as exc:
                raise SessionSwitchWhileRunningError(
                    f"Agent '{self.agent_id}' cannot switch from session '{previous_id}' "
                    f"to '{session_id}' while its event loop is running: the bus refused "
                    f"a subscription to 'session.{session_id}' ({exc}). Switching would "
                    "leave the agent reporting a session whose events it cannot receive, "
                    "so nothing was changed. Address that session without switching by "
                    "passing session_id to get_session, reset_session, load_history, "
                    "persist_session or hydrate_session."
                ) from exc
            for event in stranded:
                attributed_to = event.session_id or previous_id
                self._stranded_event_counts[attributed_to] = (
                    self._stranded_event_counts.get(attributed_to, 0) + 1
                )
            if stranded:
                logger.warning(
                    "Agent %s discarded %d queued event(s) for session '%s' on switching "
                    "to '%s'; see stranded_event_counts",
                    self.agent_id,
                    len(stranded),
                    previous_id,
                    session_id,
                )

        live = self._live_session(session_id)
        self._context = self._context.model_copy(update={"session_id": session_id})
        return live.to_state(session_id, self._config.agent_id)

    def load_history(
        self,
        messages: list[ChatMessage] | tuple[ChatMessage, ...],
        turn_counter: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """Hydrate one session's messages and turn counter from a persisted session.

        `session_id` defaults to the active session, so the existing two-argument calls
        keep their meaning.

        The values are routed through `SessionState` so they are **validated here**, at
        the call that supplied them. Writing straight into the `_LiveSession` dataclass
        accepted `turn_counter="9"` — a `slots` dataclass performs no validation — after
        which every later `get_session`, `reset_session` or `persist_session` raised
        `ValidationError` at a call site that had done nothing wrong.

        **This is one of two doors in a single family, named in full in the
        `uclone_x.agent.session` module docstring under "Validation is a property of the
        constructor, not of the type".** The other is `model_copy(update=...)` on the
        frozen `SessionState` itself, closed in `SessionState.with_messages` and backed
        by `SessionStore.save`'s pre-write revalidation. Validation added to
        `with_messages` alone cannot see *this* path, because this path never goes
        through the model; validation added here alone cannot see *that* one. Neither
        docstring used to mention the other, so a reader of either could not tell the
        family existed — read the module docstring for the whole of it before changing
        either door.

        Raises:
            SessionMutationDuringTurnError: If a turn is in flight on the target session.
            ValidationError: If `messages` or `turn_counter` are not of the declared type.
        """
        sid = self._effective_session_id(session_id)
        # "load history into", not "hydrate": the label is interpolated into the refusal,
        # and telling a caller that called `load_history` that it "cannot hydrate" names
        # something other than what it did. Same species as the `switch_session` message
        # that named a remedy which did not exist.
        self._refuse_session_mutation_during_turn(sid, "load history into")
        live = self._live_session(sid)
        state = live.to_state(sid, self._config.agent_id).with_messages(
            messages,
            turn_counter=live.turn_counter if turn_counter is None else turn_counter,
        )
        # The caller composed these messages, so their anchor is not this agent's to
        # re-resolve on a later turn (#1081). See `_anchor_is_stale`.
        replaced = _LiveSession.from_state(state, anchor_provenance=_AnchorWriter.CALLER)
        # The snapshots carry over, and so must the bodies they name that are not written
        # yet: dropping them left the next save naming bodies that were never stored.
        replaced.pending_bodies = live.pending_bodies
        replaced.stored_bodies = live.stored_bodies
        self._sessions[sid] = replaced

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
        """The session as it stands before a turn, for `roll_back_turn` to return to (#1423).

        A caller that commits a turn together with state of its own -- a room commits the
        seat's session with the transcript row and the seat's `last_seen_seq` -- takes this
        before the turn and hands it back when the turn does not commit. The checkpoint is
        the session's own frozen shape, so nothing the turn does can reach into it.

        Raises:
            SessionMutationDuringTurnError: If a turn is already in flight on the session:
                a checkpoint taken mid-turn would restore half of it.
        """
        sid = self._effective_session_id(session_id)
        self._refuse_session_mutation_during_turn(sid, "checkpoint")
        live = self._live_session(sid)
        live.last_turn_tool_calls = []
        return live.to_state(sid, self._config.agent_id)

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        """Return a session's conversation to `checkpoint`; the number of messages dropped.

        A turn that fails or is stopped leaves what it appended -- the prompt, a step's
        reply, tool results -- in the conversation, and the next attempt then stacks a
        second prompt on the first (#1423). This undoes that, and only that:

        * **The messages go back.** The conversation becomes the checkpoint's again. A
          turn that compacted before it failed rewrote the prefix too, and that
          compaction was the undone turn's own work, so it goes back with the rest; the
          event says so (`prefix_rewritten`) rather than leaving it to be inferred.
        * **The record stays.** The turn counter, the context snapshots and the request
          chain are kept, so the next request's `REQUEST_CONTEXT` records the rewind as a
          shorter kept prefix, and the log keeps every event the undone turn wrote.
          A `TURN_ROLLED_BACK` event is queued beside them, naming `reason`.
        * **Side effects are not undone.** A plan update, a memory write or a file a tool
          wrote happened; this is a statement about the conversation, not the world.
        * **The next attempt is told what was called.** The undone turn's tool calls are
          named on the event (`undone_tool_calls`) and stated in the next turn's context
          (`_undone_attempt_section`), so a retry can see that a call already ran (#1495).

        Raises:
            SessionMutationDuringTurnError: If a turn is in flight on the session.
        """
        sid = checkpoint.session_id
        self._refuse_session_mutation_during_turn(sid, "roll back a turn in")
        live = self._live_session(sid)
        kept = len(checkpoint.messages)
        prefix_rewritten = tuple(live.messages[:kept]) != checkpoint.messages
        dropped = len(live.messages) - kept if not prefix_rewritten else len(live.messages)
        if dropped or prefix_rewritten:
            live.messages = list(checkpoint.messages)
            live.updated_at = _now_iso()
        # What the undone turn called leaves the conversation with it, and the next
        # attempt is told in its turn context instead (#1495): the calls' effects stay.
        undone = live.last_turn_tool_calls
        live.last_turn_tool_calls = []
        live.undone_tool_calls.extend(undone)
        live.undone_tool_calls_shown = False
        self._pending_durable_events.append(
            {
                "type": "TURN_ROLLED_BACK",
                "turn_index": live.turn_counter,
                "reason": reason,
                "restored_message_count": kept,
                "dropped_message_count": dropped,
                "prefix_rewritten": prefix_rewritten,
                "undone_tool_calls": [
                    {"tool_call_id": call.id, "name": call.name} for call in undone
                ],
                "session_id": sid,
            }
        )
        return dropped

    def reset_session(self, session_id: str | None = None) -> SessionState:
        """Purge a session back to its seeded state and zero its turn counter.

        Returns the reset state, so a caller can persist or report it without a second
        read. Delegates to `SessionState.reset` — the Core's single reset semantics —
        rather than re-seeding here, which is what previously produced three
        implementations that disagreed.

        Two invariants this must preserve, and only one of them is automatic:

        * **The system prompt is re-seeded from `config.system_prompt`.** Nothing else
          recomposes it. `_prepare_turn_messages` composes only the per-turn sections
          around it; the system prompt is written into history once in `__init__`. A
          reset that merely cleared messages would silently strip the agent's
          instructions.
        * **The per-turn sections need no re-seeding** -- the P7 asserted invariants,
          skills and workspace joined to the system turn, and the plan and memory in the
          `[Turn Context]` block at the tail -- because `_prepare_turn_messages`
          recomposes them every turn and none is ever stored in history.

        History is not the only thing a turn accumulates. `_pending_durable_events` is
        appended to by every turn and drained only by `persist_session`, so before #670 it
        carried a reset session's events into the next one: measured 10 → 16 entries across
        two turns with a reset between them. Harmless while nothing drains it, and not
        harmless the moment a store is wired — the queue would then persist events from a
        session that was explicitly ended. It matters for evaluation in particular:
        `evaluation/answerer.py` resets between problems precisely so a score cannot depend
        on problem order, and that guarantee held for history and not for this queue. The
        rule the test pins is the general one: **after a reset, every per-session
        accumulator is empty**, not merely the history.

        The queue is agent-wide, so the clear has to be scoped or it destroys work that
        belongs to a session the caller did not name. Entries carried no session id, and
        the first version of this fix cleared the whole queue: `reset_session("sB")` on a
        session that had never run a turn wiped the *active* session's six unpersisted
        events and zeroed its step count, reachable straight from `ui/app.py`'s
        clear-history path. Deleting an audit queue is not the conservative direction —
        mis-filed events can be repaired by hand and deleted ones cannot. So every turn now
        stamps `session_id` onto the events it produces (one place, at the `extend`), and
        the clear here keeps everything that is not this session's.

        `_run_steps` and the cached compactor are scoped the same way, for the same reason:

        * `_run_steps` is a single agent-wide counter for the run in flight, and the run in
          flight belongs to the active session. Zeroing it while resetting some *other*
          session drops the budget bar `ui/app.py` renders to 0/50 for a run that is still
          going. It is zeroed only when the reset names the active session — where it is
          already zeroed at the top of the next `execute_turn` anyway, so the only window
          this changes is between a reset and that turn, and there it makes the reported
          number honest.
        * `_session_compactors[sid]` is evicted (#670 review). The cached `ContextCompactor`
          carries `_superseded_ledger_count` and `_supersession_reasons` across the reset,
          and those are rendered into prose the model reads — so a reset-then-reused session
          was told about supersessions from the conversation that was thrown away. It is
          rebuilt lazily by `_session_compactor`, so eviction costs nothing.
        * `_loaded_skills` is cleared when resetting the active session (#676). Skills
          are loaded into prompt context during a conversation via `LoadSkillTool`; without
          clearing, a skill loaded in one session leaks into the next, breaking problem
          order-independence in evaluations and session capability isolation.

        Not covered: an *injected* compactor is shared across sessions and never enters
        this dict, so that configuration still carries the model-visible ledger state
        across a reset. Eviction cannot reach it, and clearing a caller's own object is
        not this method's to do.

        Note on TokenBudgetManager: `TokenBudgetManager` provides its own
        `reset_session(session_id)` that is deliberately not invoked here: token budgets
        and financial spend limits are host-level accounting bounds across an agent's
        lifetime rather than conversation scratchpads that a session reset should evade.

        Durable events are scoped by `session_id` across `persist_session` (#682),
        `delete_session` (#682), and `reset_session` (#670), ensuring events produced by one
        session never leak into another session's persisted records or survive session deletion.

        When a `SessionStore` is wired the reset is persisted, so a reset survives the
        process rather than being undone by the next hydrate.

        Raises:
            SessionMutationDuringTurnError: If a reasoning turn is in flight on the
                target session. `execute_turn` holds `_turn_lock` across the turn and
                appends as it goes, so a reset underneath it discarded the user message
                the turn was answering: the turn then completed into the reset session
                and returned `[SYSTEM, ASSISTANT]` — an answer with no question — as
                `is_completed=True` with the turn index the reset had just zeroed.
                Nothing about that was visible to either caller.
            StaleSessionWriteError: If a store is wired and another writer committed to
                this record since this agent last agreed with it (#219). The reset is
                refused *whole* — neither disk nor memory is changed — because the write
                happens before the in-memory swap. A reset that cleared memory and failed
                to persist would be the worst of both: the conversation gone here and
                intact on disk, with the next hydrate silently restoring it.
        """
        sid = self._effective_session_id(session_id)
        self._refuse_session_mutation_during_turn(sid, "reset")
        # The prompt and the stamp are passed together because they describe one anchor:
        # this agent composes the reset anchor from `effective_system_prompt`, so the
        # axis position behind it is the one in force now, and it is recorded on the very
        # write that creates it rather than on some later save (#1152).
        reset = self.get_session(sid).reset(
            system_prompt=self.effective_system_prompt,
            anchor_provenance=_persisted_anchor_provenance(self._resolved_persona()),
        )
        if self._store is not None:
            reset = self._store.save(reset)
        self._sessions[sid] = _LiveSession.from_state(
            reset, anchor_provenance=self._resolved_persona()
        )
        self._pending_durable_events = [
            event for event in self._pending_durable_events if event.get("session_id") != sid
        ]
        self._session_compactors.pop(sid, None)
        if sid == self._context.session_id:
            self._run_steps = 0
            self._loaded_skills.clear()  # reset active session skills
        if self._store is not None:
            # The event log holds the cleared conversation's tool outputs; a reset that
            # kept it would leave them on disk and continue the next conversation in the
            # same log. Last, after the record is reset, so a refused reset keeps it (#1442).
            self._store.clear_event_log(sid)
        # Stored tool results (#1422) and offloaded outputs are the cleared conversation's
        # too, and a handle into them must not resolve in the next one.
        ws_root = self._resolve_workspace_root()
        if ws_root is not None:
            cleanup_session_artifacts(artifacts_dir_for(ws_root), sid)
        return reset

    def persist_session(
        self,
        session_id: str | None = None,
        *,
        pending_events: Sequence[Any] | None = None,
    ) -> SessionState:
        """Write one session to the Core store.

        On success the live session adopts the revision the store just wrote. That is not
        bookkeeping — it is what lets the *next* turn persist at all. The store's
        compare-and-swap compares the revision a state carries against the record, so a
        working copy left holding the revision it hydrated at would be refused on every
        write after the first, and a per-turn `persist_session` would stop being durable
        after one turn while reporting nothing.

        Raises:
            SessionStoreNotConfiguredError: If no store is wired. Returning quietly
                would leave the caller believing an in-memory session was durable,
                which is the failure mode P6 forbids reporting as a success.
            StaleSessionWriteError: If another writer committed to this record since this
                agent last agreed with it (#219). Propagated rather than resolved here:
                this agent holds messages the record does not contain and cannot know
                whether the other writer appended a turn or compacted the history away,
                so merging would be a guess. The exception carries the on-disk state, and
                `hydrate_session` is the recovery when adopting it wholesale is
                acceptable. The in-memory session is left untouched, so nothing is lost
                by the refusal — but it is now knowingly divergent from the record, which
                is the point: the divergence is reported at the moment it becomes true
                instead of being created silently by overwriting the other writer.
        """
        if self._store is None:
            raise SessionStoreNotConfiguredError(
                f"Agent '{self.agent_id}' cannot persist session state: no SessionStore "
                "was injected. Pass one as BaseAgent(store=...). Refusing rather than "
                "reporting a write that did not happen."
            )
        sid = self._effective_session_id(session_id)
        if pending_events is not None:
            events_to_persist = list(pending_events)
        else:
            events_to_persist = [
                d_event
                for d_event in self._pending_durable_events
                if d_event.get("session_id") == sid  # persist_session selects session events
            ]
        self._write_pending_bodies(sid)
        saved = self._store.save(
            self.get_session(sid), pending_events=events_to_persist if events_to_persist else None
        )
        if pending_events is None:
            self._pending_durable_events = [
                d_event
                for d_event in self._pending_durable_events
                if d_event.get("session_id") != sid  # persist_session retains other session events
            ]
        self._live_session(sid).revision = saved.revision
        return saved

    def _write_pending_bodies(self, sid: str) -> None:
        """Write the bodies `sid`'s snapshots name and the store does not hold yet.

        Called before every save of a working copy, so a record that names a body never
        reaches the disk without it.
        """
        if self._store is None:
            return
        live = self._live_session(sid)
        for digest, body in tuple(live.pending_bodies.items()):
            self._store.save_context_body(sid, digest, body)
            live.stored_bodies.add(digest)
            del live.pending_bodies[digest]

    def hydrate_session(self, session_id: str | None = None) -> SessionState | None:
        """Load one session from the Core store, replacing what is held in memory.

        Returns `None` when the store holds no record for that id, leaving the
        in-memory session untouched — an absent record is not a reason to discard a
        live conversation.

        **A record that identifies a different session raises rather than being adopted
        (#256).** This is the sharpest of the variant-id doors, and the sequence is spelled
        out step by step because it was mis-stated once and the correction makes it worse,
        not milder. Measured on a pre-fix tree, against a stored `"SessA"` holding 42 turns:

        ```
        hydrate_session("SESSA").session_id  -> 'SessA'   <- returns the OTHER id, honestly
        get_session("SESSA").session_id      -> 'SESSA'   <- the id is stamped HERE
        persist_session("SESSA")             -> writes session_id='SESSA'
        files on disk                        -> ['SessA.json']
        SessA.json's session_id on disk      -> 'SESSA'   <- erased, permanently
        ```

        So `hydrate_session` does **not** rewrite the id — its return still names `SessA`,
        which is what made the earlier account of this wrong. The rewrite happens one step
        later: `_LiveSession.from_state` keeps no id, so `_sessions["SESSA"]` renders as
        whatever key it is read under, and `get_session` stamps the asked-for id onto
        another session's history. `persist_session` then commits that **to disk**, with a
        matching revision, so the erasure is not a transient in-memory confusion — it
        destroys the record's own account of which session it is.

        It also *creates* the state where a filename and its record disagree
        (`SessA.json` holding `SESSA`), which is the tree `SessionStore.list_session_ids`
        had to be corrected to enumerate honestly. The defect manufactures its own
        aftermath.

        The refusal is in `SessionStore.load` rather than here, which is what makes this
        door and the store's own doors close together instead of one at a time.

        Raises:
            SessionStoreNotConfiguredError: If no store is wired.
            SessionIdCollisionError: If the store holds a record at this id's path that
                identifies a different session. Nothing in memory is replaced.
        """
        if self._store is None:
            raise SessionStoreNotConfiguredError(
                f"Agent '{self.agent_id}' cannot hydrate session state: no SessionStore "
                "was injected. Pass one as BaseAgent(store=...)."
            )
        sid = self._effective_session_id(session_id)
        self._refuse_session_mutation_during_turn(sid, "hydrate")
        loaded = self._store.load(sid)
        if loaded is None:
            return None
        # The record says what composed its anchor (#1152), so a restored session is
        # re-resolvable exactly when the agent composed it — and a caller's own text still
        # is not. What the record does not say is not guessed at: a stamp inferred by
        # comparing `messages[0]` against what each registered persona would compose is
        # the fragile route #1152 names, and defaulting the absence to a resolution is the
        # silent fallback P6 forbids.
        restored = _restored_anchor_provenance(loaded.anchor_provenance)
        if restored is _AnchorWriter.UNRECORDED and _has_anchor(loaded.messages):
            # Reported, not swallowed. The consequence is real and otherwise invisible:
            # a persona adopted on this session will not move the system turn the model
            # is sent, because there is no recorded position to call it stale against.
            # One `save` through `to_state` from here on records the provenance for good.
            logger.warning(
                "Session '%s' restored for agent '%s' carries a system anchor with no "
                "recorded provenance, so it will not be re-resolved when a persona is "
                "adopted (#1152). Records written before `anchor_provenance` existed read "
                "this way; persisting this session again records it.",
                sid,
                self.agent_id,
            )
        self._sessions[sid] = _LiveSession.from_state(loaded, anchor_provenance=restored)
        return loaded

    # ----------------------------------------------------------------------------------
    # Context compaction (#183 requirement 3, P5)
    #
    # The P5 split this implements: the LLM layer owns the compaction algorithm and the
    # token estimate; the Core owns only persisting the compacted sequence into session
    # state and publishing the notice, because it is the component holding the publisher
    # and the store. `p5-llm-token-management.md` requires compaction to be "strictly
    # managed by the LLM layer, completely decoupled from agent business logic", while
    # `event-driven-agent-core.md` places pruning in the agent's INGESTING state and this
    # issue's requirement 3 says to wire it into `execute_turn`. Nothing here re-implements
    # a compaction decision or a token count; both are read off the compactor.
    # ----------------------------------------------------------------------------------

    def _session_compactor(self, session_id: str) -> ContextCompactorProtocol:
        """The compactor for one session.

        An injected compactor is honoured and therefore **shared** across every session
        this agent hosts. Otherwise one `ContextCompactor` is constructed per session id.

        That default is deliberate and it is the answer to a question #196 left open.
        `ContextCompactor.superseded_ledger_count` is documented as per instance, "not
        per session", and the in-band note on a replacement ledger says "(N by this
        compactor)" precisely because a per-turn instance cannot know a session-wide
        total. Under per-session ownership the instance *is* the session accumulator, so
        in the default configuration the two coincide. The note's wording is
        deliberately left alone: `ContextCompactor` is a public class whose `compact` any
        caller may drive per turn, so "by this compactor" remains the only statement true
        in every construction pattern, and tightening it here would assert a magnitude
        the class still cannot guarantee.

        `max_ledgers` is deliberately not exposed through `AgentLLMConfig`. It is
        unvalidated above (`max_ledgers=8` reproduces #196 exactly — issue #204), so a
        configuration surface for it would hand callers a documented route back to the
        unbounded growth #201 fixed. The default of 2 is used. Exposing it becomes safe
        additively once an upper bound lands.

        No summarizer is wired by default, so the default ledger is the heuristic one.
        Passing `self._llm` would make every compaction cost an extra provider call
        inside a turn; a caller wanting LLM ledgers injects a compactor carrying its own
        summarizer.
        """
        if self._injected_compactor is not None:
            return self._injected_compactor
        compactor = self._session_compactors.get(session_id)
        if compactor is None:
            compactor = ContextCompactor(
                workspace_root=self._resolve_workspace_root(),
                session_id=session_id,
            )
            self._session_compactors[session_id] = compactor
        return compactor

    def delete_session(self, session_id: str | None = None) -> bool:
        """Delete a session's record and clean up associated tool artifacts (P3)."""
        sid = self._effective_session_id(session_id)
        if sid in self._session_compactors:
            del self._session_compactors[sid]
        if sid == self._context.session_id:
            self._loaded_skills.clear()  # delete active session skills
        self._pending_durable_events = [
            e
            for e in self._pending_durable_events
            if e.get("session_id") != sid  # delete_session drops queued events
        ]

        ws_root = self._resolve_workspace_root()
        artifacts_dir = artifacts_dir_for(ws_root) if ws_root is not None else None

        if artifacts_dir is not None:
            cleanup_session_artifacts(artifacts_dir, sid)

        if self._store is not None:
            return self._store.delete(sid, artifacts_dir=artifacts_dir)
        return False

    async def compact_session(
        self,
        session_id: str | None = None,
        reason: str = "manual_on_demand",
    ) -> CompactionResult:
        """Compact one session's context and publish a `CONTEXT_COMPACTED` notice (P5).

        The compacted sequence replaces the session's messages and is persisted when a
        `SessionStore` is wired. The turn counter is untouched: compaction discards
        context, not the fact that turns happened.

        Raises:
            SessionMutationDuringTurnError: If a reasoning turn is in flight on the
                target session. Compaction rewrites the message sequence the turn is
                mid-way through building, so an explicit `compact_session` in that
                window would discard the user message being answered — the same defect
                as a mid-turn reset. Automatic compaction is exempt by construction: it
                runs *inside* `execute_turn`, before the request is built, and calls
                the private path rather than this one.
            StaleSessionWriteError: If a store is wired and another writer committed to
                this record since this agent last agreed with it (#219). The compaction
                is abandoned rather than committed, on both sides — see the ordering
                comment in `_compact_session`. This is the case #219 measured from the
                losing side: the writer whose compaction would have been overwritten now
                keeps it, and the writer holding the pre-compaction sequence is told so
                instead of destroying it.
        """
        sid = self._effective_session_id(session_id)
        self._refuse_session_mutation_during_turn(sid, "compact")
        return await self._compact_session(sid, reason)

    async def _compact_session(
        self,
        sid: str,
        reason: str,
        *,
        hold_unseen_step: bool = False,
        reader_offered: bool | None = None,
    ) -> CompactionResult:
        """Compact one session unconditionally, without the in-flight-turn guard.

        Split from `compact_session` because automatic compaction runs *inside*
        `execute_turn`, while `_turn_lock` is held — so the public method's guard would
        refuse the one caller that is allowed to compact mid-turn. That caller is safe
        for the reason the guard exists to protect: it compacts *before* a request is
        built -- at turn start, or between two steps once every tool result of the last
        step is in the history (#1422) -- so no in-flight assistant message can be
        orphaned, and it is the turn itself rather than a concurrent caller racing it.

        `hold_unseen_step` is set between two steps (#1422): the step that just ran --
        its assistant message and its tool results -- is left out of the pass and kept
        as ingested. The model has not seen those results yet; they are already held to
        the result cap, and pruning them would send the next request without what the
        model asked for, so it would ask again.

        `reader_offered` says whether this turn offers `tool_result_read`; the short form
        of a stored result names it, so it is used only when the model can call it.
        `None` asks the registry.
        """
        live = self._live_session(sid)
        compactor = self._session_compactor(sid)
        if isinstance(compactor, ContextCompactor):
            if reader_offered is None:
                reader_offered = (
                    self._tools is not None and self._tools.get(TOOL_RESULT_READ_TOOL) is not None
                )
            compactor.tool_result_reader = reader_offered

        before_messages = list(live.messages)
        tokens_before = compactor.estimate_tokens(before_messages)

        held: list[ChatMessage] = []
        to_compact = before_messages
        if hold_unseen_step:
            start = unseen_step_start(before_messages)
            if start is not None:
                to_compact, held = before_messages[:start], before_messages[start:]

        outcome = await compactor.compact(to_compact)
        compacted = (*outcome.messages, *held)
        tokens_after = compactor.estimate_tokens(compacted)

        # Persist *before* replacing what is in memory, so a failed write leaves the
        # session untouched on both sides rather than compacted in memory and whole on
        # disk. The previous order mutated `live.messages` first, so one failed `save`
        # left the process believing a 25-message session was 6 messages long while the
        # record still held 25 — and the exception carried no hint that the in-memory
        # sequence had already been discarded. Compaction is destructive and
        # unrecoverable, so it commits atomically or not at all.
        new_state = live.to_state(sid, self._config.agent_id).with_messages(
            compacted, turn_counter=live.turn_counter
        )
        if self._store is not None:
            self._write_pending_bodies(sid)  # before the record that names them
            new_state = self._store.save(new_state)
        live.messages = list(new_state.messages)
        # Adopt the revision the store wrote, for the reason spelled out in
        # `persist_session`: a working copy holding a revision the record has moved past
        # is refused on every subsequent write. It matters more here than anywhere else,
        # because a compaction that committed and then left the session unable to persist
        # would strand the compacted context in memory only — which is the exact shape of
        # the divergence #219 measured, arrived at from the other direction.
        live.revision = new_state.revision

        result = CompactionResult(
            session_id=sid,
            reason=reason,
            ledger_source=outcome.ledger_source,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_before=len(before_messages),
            messages_after=len(compacted),
            keep_recent_turns=compactor.keep_recent_turns,
            superseded_ledger_count=outcome.superseded_ledger_count,
            # Forwarded verbatim, including `None`. For an LLM-written ledger the
            # attribution belongs to the summarizer; naming `agent.core` as the server
            # of text a model produced is the substitution P6 forbids.
            provenance=outcome.provenance,
        )

        if self._budget is not None:
            self._budget.record_compaction(
                reason=reason,
                original_tokens=tokens_before,
                compacted_tokens=tokens_after,
                kept_turns=compactor.keep_recent_turns,
                session_id=sid,
            )
        await self._publish_compaction(result)
        return result

    async def _publish_compaction(self, result: CompactionResult) -> None:
        """Announce a compaction on the bus, if this agent has a publisher.

        `AgentEvent` is frozen, strict and `extra="forbid"`, so the payload carries only
        JSON-representable scalars and `provenance` rides on the typed envelope field
        rather than as a payload key by convention. Publishing is skipped silently only
        when the agent has no bus at all — a headless agent with no publisher is a
        supported configuration, not a failure.

        If publishing fails (e.g. `EventBusError`, `QueueFullError`, delivery exception),
        the error is logged and recorded in `_processing_errors` so the absorbed
        post-commit failure is observable (P6), while allowing the committed compaction
        to complete and return its valid `CompactionResult`.
        """
        if self._publisher is None:
            return
        try:
            await self._publisher.publish(
                AgentEvent(
                    type=EventType.CONTEXT_COMPACTED,
                    topic=f"session.{result.session_id}",
                    session_id=result.session_id,
                    recipient_id="",
                    priority=EventPriority.NORMAL,
                    payload={
                        "reason": result.reason,
                        "ledger_source": result.ledger_source.value,
                        "tokens_before": result.tokens_before,
                        "tokens_after": result.tokens_after,
                        "saved_tokens": result.saved_tokens,
                        "compression_ratio_pct": result.compression_ratio_pct,
                        "messages_before": result.messages_before,
                        "messages_after": result.messages_after,
                        "keep_recent_turns": result.keep_recent_turns,
                        "superseded_ledger_count": result.superseded_ledger_count,
                    },
                    provenance=result.provenance,
                    trace_id=self._tracer.trace_id,
                )
            )
        except Exception as exc:
            self._processing_errors.append(exc)
            logger.exception(
                "Agent %s failed to publish CONTEXT_COMPACTED notice for session %s; "
                "recording absorbed failure in processing_errors",
                self.agent_id,
                result.session_id,
            )

    def _should_compact_session(
        self,
        session_id: str,
        messages: Sequence[ChatMessage],
        *,
        request: LLMRequest | None = None,
    ) -> bool:
        """Determine if a session needs auto-compaction based on configured threshold (P5).

        Delegates to the compactor's predicates so the trigger decision lives strictly in
        the LLM layer per Principle 5. Given `request`, the count is of that request --
        the rendered system turn with its sections, the turn context and the tool schemas
        -- rather than of `messages` alone, which left all three out (#1422).
        """
        llm_config = self._config.llm_config
        if not llm_config.auto_compact:
            return False
        if not messages:
            return False

        compactor = self._session_compactor(session_id)
        if request is not None:
            return self._request_over_threshold(compactor, request)

        # Trigger at 70% of the window `_context_window` resolves, or at an explicit
        # threshold that is below it.
        context_limit = self._context_window()
        if context_limit is not None:
            explicit = self._explicit_threshold(context_limit)
            if explicit is not None:
                return compactor.should_compact_at(messages, explicit)
            return compactor.should_compact(messages, context_limit)

        threshold = llm_config.compaction_threshold_tokens
        if threshold <= 0:
            return False
        return compactor.should_compact_at(messages, threshold)

    def _request_over_threshold(
        self, compactor: ContextCompactorProtocol, request: LLMRequest
    ) -> bool:
        """`_should_compact_session`'s limit resolution, applied to a whole request."""
        llm_config = self._config.llm_config
        context_limit = self._context_window()
        if context_limit is not None:
            explicit = self._explicit_threshold(context_limit)
            if explicit is not None:
                return compactor.should_compact_request_at(request, explicit)
            return compactor.should_compact_request(request, context_limit)
        threshold = llm_config.compaction_threshold_tokens
        if threshold <= 0:
            return False
        return compactor.should_compact_request_at(request, threshold)

    async def _auto_compact_if_needed(
        self,
        tools: Sequence[ToolDefinition] = (),
        extra_sections: Sequence[str] = (),
        *,
        reason: str = "auto_threshold",
        hold_unseen_step: bool = False,
    ) -> CompactionResult | None:
        """Compact the active session when the request about to be sent reaches the threshold.

        Returns the result when a pass ran, else `None`.

        The count is of the request the next step will send: the history as
        `_prepare_turn_messages` renders it with `extra_sections`, and `tools` (#1422).
        The trigger delegates to `_should_compact_session`, calling the LLM layer's
        predicates.
        """
        await self._observe_context_window()
        request = LLMRequest(
            messages=tuple(self._prepare_turn_messages(extra_sections=extra_sections)),
            tools=tuple(tools),
        )
        if not self._should_compact_session(
            self._context.session_id, self._history, request=request
        ):
            return None
        # The unguarded path: this *is* the turn, so the in-flight-turn guard on the
        # public `compact_session` would refuse its own caller.
        return await self._compact_session(
            self._context.session_id,
            reason,
            hold_unseen_step=hold_unseen_step,
            reader_offered=any(d.name == TOOL_RESULT_READ_TOOL for d in tools),
        )

    def _get_active_invariants_prompt_section(
        self,
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> str:
        """Query active invariants using tier_filter='asserted' by default (P7, P8)."""
        if self._ontology is None:
            return ""
        invariants = self._ontology.get_active_invariants(domain=domain, tier_filter=tier_filter)
        if not invariants:
            return ""
        lines = ["[Active Domain Ontology Invariants]:"]
        for inv in invariants:
            rule_detail = (
                inv.rule_expression
                or (
                    f"{inv.predicate} == {inv.object_value}"
                    if inv.predicate and inv.object_value
                    else inv.predicate
                )
                or inv.description
                or inv.name
            )
            lines.append(f"- Rule ({inv.tier.value}): {inv.name} -> {rule_detail}")
        return "\n".join(lines)

    def _get_active_skills_prompt_section(self) -> str:
        """Return progressive disclosure prompt section listing approved skills (P9)."""
        if self._skills is None:
            return ""
        active_skills = [
            s for s in self._skills.list_skills() if s.manifest.status == SkillStatus.ACTIVE
        ]
        if not active_skills:
            return ""
        active_skills.sort(key=lambda s: s.manifest.name)
        lines = [
            "[Available Approved Skills]",
            "The following modular skills are approved and available for on-demand use. "
            "To view full procedural instructions for any skill, invoke the 'load_skill' tool.",
        ]
        for s in active_skills:
            desc = (s.manifest.description or "").strip() or "No description provided."
            lines.append(f"- {s.manifest.name}: {desc}")
        return "\n".join(lines)

    def _get_workspace_prompt_section(self) -> str | None:
        """Tell the model where its file tools resolve paths, when it holds any.

        Without it the model knows only that it works "in the user's workspace", so it
        invents a relative path for a folder it was told about by name and cannot tell a
        folder that is absent from one that is out of bounds.
        """
        allowed = self._config.allowed_tools
        # A name in the list is a permission, not a tool: every persona is now permitted
        # the read-only file tools (#1402), including on a registry that has none. The
        # section is only true when one of them is actually there to be called.
        if allowed and not any(
            name in _FILE_TOOL_NAMES and self._tools is not None and self._tools.get(name)
            for name in allowed
        ):
            return None
        workspace = self._resolve_workspace_root()
        if workspace is None:
            return None
        lines = [
            "[Workspace]",
            f"File tools resolve relative paths against the workspace folder: {workspace}",
        ]
        read_roots = self._config.read_roots
        if read_roots:
            lines.append(
                "The file tools may also read (never write) these folders outside it, by "
                "absolute path:"
            )
            lines.extend(f"- {root.resolve()}" for root in read_roots)
        lines.append(
            "The file tools refuse any other path. When a folder or file is not found, "
            "list the folder that should contain it before telling the user it is missing."
        )
        return "\n".join(lines)

    def _get_active_plan_prompt_section(self) -> str | None:
        """Format the current interactive execution plan for injection into the system prompt."""
        plan = self.current_plan
        if not plan:
            return None

        lines = [f"### Active Execution Plan: {plan.title}", f"Status: {plan.status.upper()}", ""]

        for step in plan.steps:
            box = "[x]" if step.completed else "[ ]"
            lines.append(f"{step.index}. {box} {step.description}")
            if step.verification:
                lines.append(f"   Verification: {step.verification}")

        return "\n".join(lines)

    def _record_context_snapshot(self, req: LLMRequest, layers: RequestLayers) -> str:
        """Record what `req` carries besides the conversation; return the snapshot's id.

        Appends a `ContextSnapshot` to the active session unless the last one already says
        the same, so a turn adds one, and a step adds another only when its tools, identity,
        slow context, turn context or model settings differ. The large bodies wait on
        the session until the next save writes them, once each, ahead of the record.
        """
        session = self._active_session
        tools_body = serialize_tools(req.tools)
        bodies = {
            content_digest(tools_body): tools_body,
            content_digest(layers.identity): layers.identity,
            content_digest(layers.slow_context): layers.slow_context,
            content_digest(layers.turn_context): layers.turn_context,
        }
        candidate = ContextSnapshot(
            turn_index=self._turn_counter,
            tools_digest=content_digest(tools_body),
            identity_digest=content_digest(layers.identity),
            slow_context_digest=content_digest(layers.slow_context),
            system_message=layers.system_message,
            turn_context_digest=content_digest(layers.turn_context),
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            auto_compact=req.auto_compact,
            compaction_threshold_tokens=req.compaction_threshold_tokens,
        )
        for digest, body in bodies.items():
            if digest not in session.stored_bodies:
                session.pending_bodies[digest] = body
        if not session.context_snapshots or session.context_snapshots[-1] != candidate:
            session.context_snapshots.append(candidate)
        return candidate.snapshot_id

    def _request_context_fields(
        self, step: int, req: LLMRequest, layers: RequestLayers
    ) -> dict[str, Any]:
        """The `REQUEST_CONTEXT` event's fields for `req`: its snapshot, and what it added.

        The event names the snapshot that holds the tools, identity, slow context, turn
        context and model settings, and records the conversation as a delta on the
        previous request of this session -- the previous step, or the last step of the
        previous turn. A turn's first step used to record its whole request, so every turn
        re-recorded the conversation so far (#1421). `rebuild_requests` reverses this.

        `request` numbers the requests of this working copy and `base_request` names the
        one this extends, so a reader can tell a gap in the log from a real extension.
        `digest` is over the whole request as sent, before redaction.
        """
        session = self._active_session
        snapshot_id = self._record_context_snapshot(req, layers)
        conversation = [m.model_dump() for m in layers.conversation]
        kept, appended = _request_context_delta(session.last_conversation, conversation)
        base_request = session.last_request
        request_number = 1 if base_request is None else base_request + 1
        session.last_conversation = conversation
        session.last_request = request_number
        return {
            "step": step,
            "snapshot": snapshot_id,
            "request": request_number,
            "base_request": base_request,
            "message_count": len(req.messages),
            "kept_message_count": kept,
            "appended_messages": appended,
            "digest": messages_digest([m.model_dump() for m in req.messages]),
        }

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
        context (the request-layering design, §5.5), and `_turn_context_block`
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
        """Construct turn layers: invariants and skills in the system turn, volatile state at the tail (P7, P8, P9).

        Returns the layers apart, so the turn can record them in a `ContextSnapshot`;
        `assemble_request_messages` puts them together, for the request and for a rebuild
        of it from the record alike (#1421).

        **Only sections that hold still across a conversation join the system turn** --
        the asserted invariants and the approved skills. The plan, cross-session memory
        and `extra_sections` change inside a conversation and travel at the tail of the
        request instead, so a change to them does not discard the cached prefix; see
        `_turn_context_block`.

        The anchored system turn is **re-framed for the configured model family on every
        turn** rather than trusted as `self._history[0]` (#921). `hot_reload_llm` moves
        `llm_config.model_name`, so `effective_system_prompt` starts answering for the
        new family immediately while the seeded anchor still carries the old family's
        steerability framing; the wire call took the anchor, so the property and the
        message actually sent disagreed, silently (P6).

        Resolved here rather than by rewriting the anchor in place, for two reasons that
        the in-place rewrite cannot reach:

        * **Sessions this agent has not materialised yet.** `_live_session` seeds on
          first touch and `load_history` hydrates from the store, both of which can
          happen *after* a `hot_reload_llm` call. A refresh that walks `self._sessions`
          at reload time reframes only what is live at that instant; a session restored
          from a record persisted under the old model stays stale for the rest of its
          life. Resolving on the turn covers every session on the only path that matters.
        * **The stored history keeps saying what was actually anchored.** The anchor is
          persisted, so overwriting it would edit the record of earlier turns to claim
          framing those turns never carried, and it is unrecoverable once saved.

        `adapt_system_prompt` is a substitution *from* the canonical policy, so an
        operator's own prompt — one that embeds no framing this codebase knows — comes
        back byte-identical and is never replaced by the resolved default.

        **The persona axis is resolved here too, against the axis position the anchor was
        composed under** (#1081). Adopting a persona changes which prompt the agent is
        supposed to be sending, and a session anchored before that adoption carries the
        previous one, so `effective_system_prompt` and the wire described different
        personas for the same agent. It is resolved on the turn for the same two reasons
        the model family is, and `_anchor_is_stale` decides it per session by comparing
        that session's `anchor_provenance` against what the axis resolves to now.

        Reading it off the *session* rather than off the agent is not a detail. The eleven
        constructions in the PR body were measured against two agent-wide flags before
        this one, and the four regressions between them — C1, C2, C8, C9 — are all the
        same shape: one flag answering for a session whose anchor a caller composed and a
        session the agent seeded, which are different answers. Per session it also reaches
        a session seeded on first touch, or one hydrated after the persona was adopted,
        which a refresh walking `self._sessions` at set time structurally cannot.

        Clearing an adopted persona is resolved by the same comparison rather than by a
        case of its own: `_system_prompt_base` falls back to `config.system_prompt`, so an
        anchor composed under the persona just dropped is recomputed from configuration
        rather than salvaged.
        """
        sections: list[str] = []
        invariants_section = self._get_active_invariants_prompt_section(tier_filter="asserted")
        if invariants_section:
            sections.append(invariants_section)
        skills_section = self._get_active_skills_prompt_section()
        if skills_section:
            sections.append(skills_section)

        workspace_section = self._get_workspace_prompt_section()
        if workspace_section:
            sections.append(workspace_section)

        turn_sections: list[str] = [section for section in extra_sections if section]
        plan_section = self._get_active_plan_prompt_section()
        if plan_section:
            turn_sections.append(plan_section)

        if self._memory is not None:
            memory_section = self._memory.format_prompt_section()
            if memory_section:
                turn_sections.append(memory_section)

        messages = list(self._history)
        combined_section = "\n\n".join(sections)
        turn_context = _turn_context_block(turn_sections)

        anchored_turn = _has_anchor(messages)
        if anchored_turn:
            # The anchor is re-framed whether or not there is a section to inject: the
            # framing has to track the model either way, and returning `self._history`
            # untouched on the no-sections path is how #921 stayed live for an agent with
            # no ontology, skills, plan or memory wired up.
            anchored = messages[0].content or ""
            base_sys = adapt_system_prompt(
                self._system_prompt_base()
                if self._anchor_is_stale(self._active_session)
                else anchored,
                self._config.llm_config.model_name,
            )
        else:
            # **No anchor at all: the turn sends what `effective_system_prompt` reports**
            # (#1091). A session hydrated through `load_history` from rows whose first is
            # not a `SYSTEM` turn, `load_history([])`, and a session seeded under an empty
            # `config.system_prompt` before a persona was adopted all reach here. This
            # branch used to send the sections alone, or no system turn at all, while
            # `effective_system_prompt` named the persona's or the configured prompt.
            #
            # Resolved here, on the turn, rather than by the other two answers #1091
            # names. *Seeding an anchor at hydration* would write into the record a system
            # turn the caller did not supply and earlier turns were never sent (#1078), and
            # cannot reach a seeded session that simply had no prompt until a persona
            # arrived. *Refusing the hydration* breaks every caller that loads a
            # user-first transcript today. Synthesising is also not #1081's hazard of
            # discarding a caller's anchor: there is no caller-composed system turn to
            # discard, so nothing a caller supplied is replaced, and history is left as
            # loaded. An empty resolution synthesises nothing, exactly as
            # `SessionState.seed` writes no empty `SYSTEM` turn.
            base_sys = self.effective_system_prompt

        # One composition for both branches, so a synthesised turn carries the sections in
        # exactly the order and spacing an anchored one does.
        resolved = compose_system_message(base_sys, combined_section)
        return RequestLayers(
            identity=base_sys,
            slow_context=combined_section,
            system_message=anchored_turn or bool(resolved),
            conversation=tuple(messages[1:] if anchored_turn else messages),
            turn_context=turn_context,
        )

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
        progress: _StreamProgress | None = None,
    ) -> ModelResponse:
        """One model invocation, with the token budget checked before and charged after.

        Pre-flight before spend is committed, reconciliation after: a call already in
        flight when the budget was open is allowed to finish, and its usage is charged
        before the next step is admitted (`dynamic-persona-interface.md` §4.1). Both live
        here rather than at the call site so every agent step is covered by construction,
        including the steps a tool-using turn adds.

        Nothing here catches a budget error: a step that answered is booked, and a ceiling
        it reaches refuses the *next* step, at the pre-flight check above, with the reason
        on the refusal.

        A stream that fails before it finishes raises `LLMStreamInterruptedError`, after
        booking what the partial stream spent; it is not retried through `generate` (#938).
        A stream cancelled before it finishes books the same way and re-raises the
        `CancelledError` unchanged. `progress`, when given, receives the streamed content,
        thinking and booked usage as they arrive, so a caller can record a failed call.
        """
        if self._budget is not None:
            self._budget.enforce_budget(self._context.session_id, provider=llm.provider_name)

        if stream_callback is not None and hasattr(llm, "stream"):
            # What was asked for: the request's model, else the one the connector says it
            # sends a request naming none to. `None` when neither is knowable -- never the
            # literal "default", which is a placeholder no provider serves and which the
            # room then showed as `ollama:default` while `OLLAMA_MODEL` answered (#1447).
            requested_model = (
                _named_model(req.model)
                or _named_model(getattr(llm, "_default_model", None))
                or _named_model(getattr(llm, "model", None))
            )
            # What served it: whatever the stream itself reports, like `generate` reads the
            # response body. The last chunk that names one wins.
            served_model: str | None = None
            if progress is None:
                progress = _StreamProgress()
            content_chunks = progress.content_chunks
            thinking_chunks = progress.thinking_chunks
            tool_calls_list: list[ToolCallRequest] = []
            last_usage: TokenUsage | None = None
            last_finish: FinishReason | None = None
            chunks_received = 0
            # A label for the log lines below, not an attribution.
            eff_model_name = requested_model or "an unreported model"
            try:
                async for chunk in llm.stream(req):
                    chunks_received += 1
                    if chunk.model:
                        served_model = chunk.model
                        eff_model_name = chunk.model
                    if chunk.delta_thinking:
                        thinking_chunks.append(chunk.delta_thinking)
                        if len(thinking_chunks) == 1:
                            logger.info(
                                "🧠 [%s] Model %s began thinking/reasoning",
                                self.agent_id,
                                eff_model_name,
                            )
                        elif len(thinking_chunks) % 50 == 0:
                            logger.debug(
                                "🧠 [%s] Model %s thinking progress: %d tokens",
                                self.agent_id,
                                eff_model_name,
                                len(thinking_chunks),
                            )
                        res = stream_callback(
                            "status",
                            {"status": "thinking", "detail": chunk.delta_thinking},
                        )
                        if asyncio.iscoroutine(res):
                            await res
                    if chunk.delta_content:
                        if thinking_chunks and not content_chunks:
                            logger.info(
                                "💬 [%s] Model %s finished thinking (%d tokens); generating response",
                                self.agent_id,
                                eff_model_name,
                                len(thinking_chunks),
                            )
                        content_chunks.append(chunk.delta_content)
                        res = stream_callback(
                            "token",
                            {"content": chunk.delta_content},
                        )
                        if asyncio.iscoroutine(res):
                            await res
                    if chunk.tool_calls:
                        tool_calls_list.extend(chunk.tool_calls)
                    if chunk.usage:
                        last_usage = chunk.usage
                    if chunk.finish_reason:
                        last_finish = chunk.finish_reason
            except asyncio.CancelledError:
                # Stopped by the caller (the room's Stop). What streamed was served and is
                # booked like an interrupted stream's, then the cancellation propagates
                # unchanged: it is never converted into a turn error.
                cancelled_usage = self._book_partial_stream(
                    llm,
                    req,
                    last_usage,
                    served_model or requested_model,
                    chunks_received,
                    tool_calls_list,
                    content_chunks,
                )
                progress.usage = cancelled_usage
                raise
            except Exception as stream_err:
                # The stream stopped before it finished, and the turn fails (#938). This
                # used to re-request the step through `llm.generate` under `PRIMARY`
                # provenance: paid twice, the whole reply replayed to the listener after the
                # partial one, and nothing in band saying so. Nothing is retried now. The
                # partial reply and any tool call it carried are discarded -- a call's
                # arguments may be truncated, and the step that asked for it never ended --
                # and the partial stream is booked: the provider's count if a chunk carried
                # one, an estimate of what arrived if any chunk did, and nothing if none
                # did, since then nothing shows the provider served the request. Design:
                # the room UI design document §6.7 [#938].
                known_model = served_model or requested_model
                progress.usage = self._book_partial_stream(
                    llm,
                    req,
                    last_usage,
                    known_model,
                    chunks_received,
                    tool_calls_list,
                    content_chunks,
                )
                source = (
                    f"{llm.provider_name}/{known_model}"
                    if known_model
                    else f"{llm.provider_name} (model not reported)"
                )
                interrupted = LLMStreamInterruptedError(
                    f"The streamed reply from {source} was "
                    f"interrupted after {chunks_received} chunk(s) by "
                    f"{type(stream_err).__name__}: {stream_err}. The turn failed and was "
                    f"not retried; the partial reply and {len(tool_calls_list)} tool call(s) "
                    "it carried were discarded unexecuted.",
                    provider=llm.provider_name,
                    model=known_model,
                    chunks_received=chunks_received,
                    discarded_tool_calls=len(tool_calls_list),
                    partial_content="".join(content_chunks),
                )
                raise interrupted from stream_err
            else:
                full_content = "".join(content_chunks)
                full_thinking = "".join(thinking_chunks) if thinking_chunks else None
                # A request that named no model resolved, on the provider's side, to the one
                # the stream reports; that is what was asked for, not a substitution. A
                # stream that names none was served by what was asked for, as `generate`
                # assumes when the body names none. Neither known: `None`, stated (#1447).
                provenance = Provenance.primary(
                    provider=llm.provider_name,
                    model=requested_model or served_model,
                    served_model=served_model,
                )
                model_name = served_model or requested_model or "unknown"
                if last_usage is None:
                    # The stream ended without the provider's count. The same step served
                    # by `generate` -- which is what runs when nobody is listening -- would
                    # carry one, so this is where a listener could change what the ceiling
                    # sees (#916). The figure is labelled an estimate in-band, and
                    # `record_usage` books it like a count.
                    last_usage = self._estimate_stream_usage(
                        llm, req, served_model or requested_model, full_content, tool_calls_list
                    )
                finish_reason = last_finish or (
                    FinishReason.TOOL_CALLS if tool_calls_list else FinishReason.STOP
                )
                resp = ModelResponse(
                    content=full_content,
                    thinking=full_thinking,
                    tool_calls=tuple(tool_calls_list),
                    usage=last_usage,
                    finish_reason=finish_reason,
                    model_name=model_name,
                    provenance=provenance,
                )
        else:
            resp = await llm.generate(req)
            if stream_callback is not None and resp.content:
                res = stream_callback("token", {"content": resp.content})
                if asyncio.iscoroutine(res):
                    await res

        if self._budget is not None:
            self._budget.record_usage(self._context.session_id, resp.usage)
        return resp

    def _book_partial_stream(
        self,
        llm: LLMProviderProtocol,
        req: LLMRequest,
        last_usage: TokenUsage | None,
        known_model: str | None,
        chunks_received: int,
        tool_calls: Sequence[ToolCallRequest],
        content_chunks: Sequence[str],
    ) -> TokenUsage | None:
        """Book what a stream that stopped early spent, and return that figure (#938).

        The provider's count if a chunk carried one, an estimate of what arrived if any
        chunk did, and nothing if none did, since then nothing shows the provider served
        the request.
        """
        partial_usage = last_usage
        if partial_usage is None and chunks_received:
            partial_usage = self._estimate_stream_usage(
                llm, req, known_model, "".join(content_chunks), tool_calls
            )
        if self._budget is not None and partial_usage is not None:
            self._budget.record_usage(self._context.session_id, partial_usage)
        return partial_usage

    @staticmethod
    def _estimate_stream_usage(
        llm: LLMProviderProtocol,
        req: LLMRequest,
        model_name: str | None,
        content: str,
        tool_calls: Sequence[ToolCallRequest],
    ) -> TokenUsage:
        """A labelled stand-in for the count a stream did not send (#916, #938).

        The estimate a connector's `generate` makes for a count its provider did not report
        (`connectors/base.py::resolve_token_counts`), so a watched step and a headless step
        book the same figure for the same request and reply (#980). The input is the
        request's messages and tool definitions; the output is the reply text and tool calls
        that actually arrived -- the whole reply for a stream that ended, the partial one for
        a stream that failed. It was `len // 4` over characters, with no message framing, no
        tool definitions and no reply tool calls, and so differed from the headless figure on
        ASCII too. The estimator's error is measured in the room UI design document §6.7.
        """
        in_tokens = estimate_request_tokens(req)
        out_tokens = estimate_reply_tokens(content, tool_calls)
        return TokenUsage(
            provider=llm.provider_name,
            model=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            count_source=TokenCountSource.ESTIMATE,
        )

    async def _invoke_step_model(
        self,
        llm: LLMProviderProtocol,
        req: LLMRequest,
        step: int,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None,
        durable_events: list[dict[str, Any]],
        step_usages: list[TokenUsage],
    ) -> ModelResponse:
        """`_invoke_model`, recorded as one `MODEL_RESPONSE` whether it returns or raises.

        On an exception -- cancellation included -- the event carries what had streamed
        (content, thinking) and the usage the budget was charged for it, then the exception
        is re-raised unchanged. That charged usage also joins `step_usages`, so the turn's
        `usage` and the budget agree on an interrupted step (turn inspection design §4.2).
        """
        started_at = _now_iso()
        is_streamed = stream_callback is not None and hasattr(llm, "stream")
        progress = _StreamProgress()
        try:
            resp = await self._invoke_model(
                llm, req, stream_callback=stream_callback, progress=progress
            )
        except (Exception, asyncio.CancelledError) as invoke_exc:
            ended_at = _now_iso()
            partial_content = getattr(invoke_exc, "partial_content", None)
            if not isinstance(partial_content, str):
                partial_content = progress.content if progress.content_chunks else None
            err_model = _named_model(getattr(invoke_exc, "model", None)) or _named_model(req.model)
            if progress.usage is not None:
                step_usages.append(progress.usage)
            durable_events.append(
                {
                    "type": "MODEL_RESPONSE",
                    "turn_index": self._turn_counter,
                    "step": step,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "streamed": is_streamed,
                    "content": partial_content or "",
                    "thinking": progress.thinking,
                    "tool_calls": [],
                    "finish_reason": None,
                    "model_name": err_model,
                    "usage": (
                        progress.usage.model_dump(mode="json")
                        if progress.usage is not None
                        else None
                    ),
                    "error": {
                        "type": type(invoke_exc).__name__,
                        "message": str(invoke_exc),
                        "partial_content": partial_content,
                    },
                    **_model_name_reason(
                        err_model,
                        "the call failed before any model was reported, and the request named none",
                    ),
                }
            )
            raise
        ended_at = _now_iso()
        # Read defensively: a test double may hand back something other than a
        # `ModelResponse`, and recording the step must not be what fails the turn.
        usage = getattr(resp, "usage", None)
        if not isinstance(usage, TokenUsage):
            usage = None
        if usage is not None:
            step_usages.append(usage)
        finish_reason = getattr(resp, "finish_reason", None)
        finish_str = getattr(finish_reason, "value", None)
        # `ModelResponse.model_name` defaults to "unknown"; that names no model either.
        served_name = _named_model(getattr(resp, "model_name", None))
        model_name = (served_name if served_name != "unknown" else None) or _named_model(req.model)
        durable_events.append(
            {
                "type": "MODEL_RESPONSE",
                "turn_index": self._turn_counter,
                "step": step,
                "started_at": started_at,
                "ended_at": ended_at,
                "streamed": is_streamed,
                "content": getattr(resp, "content", "") or "",
                "thinking": getattr(resp, "thinking", None),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": unwrap_immutable(tc.arguments),
                    }
                    for tc in getattr(resp, "tool_calls", ()) or ()
                ],
                "finish_reason": finish_str,
                "model_name": model_name,
                "usage": usage.model_dump(mode="json") if usage is not None else None,
                "error": None,
                **_model_name_reason(
                    model_name, "the provider reported no model, and the request named none"
                ),
            }
        )
        return resp

    async def execute_turn(
        self,
        input_data: str | AgentEvent,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        caller_turn_id: str | None = None,
    ) -> TurnResult:
        """Execute a single reasoning turn with serialized execution lock (P4, Issue #60).

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
        llm = self._llm
        if llm is None:
            configured = self._config.llm_config.model_name
            wanted = (
                f"model '{configured}' is configured" if configured else "no model is configured"
            )
            raise LLMConnectorNotConfiguredError(
                f"Agent '{self.agent_id}' cannot execute a reasoning turn: {wanted} but no "
                "LLM connector was injected. Pass one as BaseAgent(llm=...). Refusing "
                "rather than substituting a canned reply for an answer nothing computed."
            )

        async def _emit_stream(event_name: str, data: dict[str, Any]) -> None:
            if stream_callback is not None:
                try:
                    res = stream_callback(event_name, data)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    logger.debug("Error in stream_callback", exc_info=True)

        async with self._turn_lock:
            await _emit_stream(
                "status",
                {"status": "thinking", "detail": "Analyzing user prompt..."},
            )
            correlation_id = input_data.event_id if isinstance(input_data, AgentEvent) else None
            live_active_skills = (
                tuple(
                    sorted(
                        s.manifest.name
                        for s in self._skills.list_skills()
                        if s.manifest.status == SkillStatus.ACTIVE
                    )
                )
                if self._skills is not None
                else ()
            )
            loaded_skills_snapshot = tuple(sorted(self._loaded_skills))
            # A turn that failed leaves the agent in ERROR (see the handler below), whose
            # only legal successors are IDLE and TERMINATED. Clearing it here is what
            # stops the *next* turn raising `InvalidStateTransitionError` out of the
            # `transition_to(INGESTING)` below — which sits outside the try and so could
            # never be reported as a failed `TurnResult` (#136b). Recovering at the start
            # of the next turn rather than at the end of the failed one keeps ERROR
            # observable on `state`/`context.current_state` in between.
            if self._state is AgentState.ERROR:
                self.transition_to(AgentState.IDLE)

            self._turn_counter += 1
            self.transition_to(AgentState.INGESTING)

            # The `try` opens here, not after the REASONING transition. Ingestion and
            # automatic compaction used to sit outside it, which is the #136b class the
            # comment above names: a compaction that raised — one failed `SessionStore`
            # write is enough, in the default no-summarizer configuration — escaped
            # `execute_turn` raw, left the state on INGESTING rather than ERROR, burned
            # the turn counter, published no notice, and returned no `TurnResult`. The
            # identical failure eleven lines lower got ERROR, an attributed
            # `PROVIDER_FAILOVER` and `TurnResult(is_completed=False, provenance=...)`.
            # Read once, at the top of the turn, and stamped onto every event this turn
            # produces in the `finally` below.
            #
            # Reading it at the end would be equivalent today: `_turn_lock` is held across
            # the whole body including the `finally`, and `switch_session` refuses while a
            # turn is running, so line 871 cannot reassign `_context.session_id` underneath
            # us. (An earlier version of this comment claimed a mid-turn switch was legal.
            # It is not — review of #670 demonstrated `SessionMutationDuringTurnError`.)
            # Capturing at the top is kept because it does not depend on that lock staying
            # where it is: the two are equivalent, and only one of them stays correct if
            # the locking changes.
            turn_session_id = self._context.session_id
            # Zeroed here rather than beside the loop: two returns sit above that point --
            # a PRE_TURN hook block and a pre-loop exception -- and both reported the
            # *previous* turn's count until this moved (review of #702). The eval path was
            # correct only by accident, because it resets the session between problems.
            self._run_steps = 0
            # Hoisted above the `try` with the counter, for the same reason: the `finally`
            # that writes TURN_END reads them, and a turn that raised before the loop left
            # them unbound.
            tool_executions: list[ToolExecutionRecord] = []
            # Accumulated across steps, not just the last one. `tool_executions`
            # already accumulated; `tool_calls` did not, so a turn that used a tool and
            # then answered reported *no* tool calls — the final step has none. The
            # surface reads this field to show what ran.
            # Hoisted for the error handlers, which report what ran before the failure
            # (#1366): returning `()` there told every reader an errored turn used no
            # tools, although earlier steps may have run them and written files.
            all_tool_calls: list[ToolCallRequest] = []
            # The trailing calls a refused step withheld; already stated as undone (#1509).
            withheld_calls = 0
            # True only while a step's tools are running and their records have not yet
            # reached `tool_executions`. A failure in that window may have lost records of
            # calls that ran, so the result then says its list is incomplete.
            tools_unreported = False
            stop_reason: TurnStopReason = "not_started"
            step_usages: list[TokenUsage] = []
            # `caller_turn_id` is spread in by a helper: `execute_turn` sits at pyright's
            # code-flow complexity limit, and one more branch here makes it unanalysable.
            durable_events: list[dict[str, Any]] = [
                {
                    "type": "TURN_START",
                    "turn_index": self._turn_counter,
                    "agent_id": self.agent_id,
                    "at": _now_iso(),
                    **_caller_turn_id_field(caller_turn_id),
                }
            ]
            try:
                if isinstance(input_data, str):
                    content_input = input_data
                else:
                    raw_payload: Mapping[str, Any] = input_data.payload
                    raw_msg = raw_payload.get("message", raw_payload.get("content", ""))
                    content_input = str(raw_msg)

                # Execute PRE_TURN hook
                pre_turn_ctx = HookContext(
                    agent_id=self.agent_id,
                    session_id=self._context.session_id,
                    trace_id=self._tracer.trace_id,
                    event_type=HookEvent.PRE_TURN,
                    payload={
                        "input": content_input,
                        "turn_index": self._turn_counter,
                    },
                )
                pre_turn_decision = await self._hook_runner.run_hooks(
                    HookEvent.PRE_TURN, pre_turn_ctx
                )
                if pre_turn_decision.action == HookAction.BLOCK:
                    block_reason = pre_turn_decision.reason or "Blocked by pre_turn hook"
                    # Structured, so a consumer need not recognise the sentence below
                    # (#970); the `finally` records the same value on `TURN_END`.
                    stop_reason = "blocked_by_hook"
                    self.transition_to(AgentState.IDLE)
                    return TurnResult(
                        turn_index=self._turn_counter,
                        steps_taken=self._run_steps,
                        content=f"Turn execution blocked by hook: {block_reason}",
                        tool_calls=(),
                        tool_executions=(),
                        is_completed=False,
                        error=block_reason,
                        stop_reason=stop_reason,
                        correlation_id=correlation_id,
                        provenance=None,
                        persona=self.persona_name or self.persona,
                        active_skills=live_active_skills,
                        loaded_skills=loaded_skills_snapshot,
                        usage=None,
                    )
                if (
                    pre_turn_decision.action == HookAction.MODIFY
                    and pre_turn_decision.modified_payload
                ):
                    mod = pre_turn_decision.modified_payload
                    if "input" in mod:
                        content_input = str(mod["input"])
                    elif "message" in mod:
                        content_input = str(mod["message"])
                    elif "content" in mod:
                        content_input = str(mod["content"])

                user_prompt = redact_message(
                    ChatMessage(role=MessageRole.USER, content=content_input)
                )
                if not _repeats_unanswered_prompt(self._history, user_prompt):
                    self._history.append(user_prompt)
                durable_events.append(
                    {
                        "type": "USER_MESSAGE",
                        "content": content_input,
                    }
                )

                self.transition_to(AgentState.REASONING)
                await _emit_stream(
                    "status",
                    {"status": "thinking", "detail": "Formulating reasoning plan..."},
                )
                tool_defs: list[ToolDefinition] = []
                scoping_notice = ""
                if self._tools is not None:
                    for t in self.available_tools():
                        tool_defs.append(
                            ToolDefinition(
                                name=t.name,
                                description=t.description,
                                parameters=t.parameters_schema,
                            )
                        )
                    # Layer 2: tool scoping. A withheld tool is a withheld capability,
                    # so the notice travels with the turn (P6); scoping in silence is
                    # indistinguishable from a registry that never held the tool.
                    if self._tool_scoper is not None:
                        scoping = await self._tool_scoper.scope_tools(
                            content_input, tuple(tool_defs)
                        )
                        tool_defs = list(scoping.selected)
                        scoping_notice = scoping.notice()

                # Kept for the turn, not just its first step: the tool list scoped above is
                # sent on every step, so the notice naming what it withheld is too. The step
                # rebuild below used to drop it: from the second step on, the scoped list
                # arrived with no word that anything was withheld.
                #
                # What an undone attempt called, stated once per retry and kept for every
                # step of it, like the notice (#1495). Cleared when the turn that showed it
                # was not rolled back: that turn is now in history and speaks for itself.
                turn_live = self._live_session(turn_session_id)
                if turn_live.undone_tool_calls_shown:
                    turn_live.undone_tool_calls.clear()
                    turn_live.undone_tool_calls_shown = False
                undone_section = _undone_attempt_section(turn_live.undone_tool_calls)
                turn_live.undone_tool_calls_shown = bool(undone_section)
                turn_extra_sections = tuple(
                    section for section in (scoping_notice, undone_section) if section
                )

                # Compact before dispatch (Requirement 3). Placed after the user message is
                # appended, so the message that may itself tip the context over the
                # threshold is counted, and after the tool list is settled, so the count
                # is of the request this step sends -- system sections, turn context and
                # tool schemas included (#1422) -- and before `_prepare_turn_layers`,
                # so the request is built from the compacted sequence.
                await self._auto_compact_if_needed(tool_defs, turn_extra_sections)

                req_layers = self._prepare_turn_layers(extra_sections=turn_extra_sections)
                turn_messages = assemble_request_messages(req_layers)
                req = LLMRequest(
                    model=self._config.llm_config.model_name or None,
                    messages=tuple(turn_messages),
                    tools=tuple(tool_defs),
                    temperature=self._config.llm_config.temperature,
                    max_tokens=self._config.llm_config.max_tokens,
                    auto_compact=self._config.llm_config.auto_compact,
                    compaction_threshold_tokens=(
                        self._config.llm_config.compaction_threshold_tokens
                    ),
                    context_window=(
                        self._config.llm_config.context_limit
                        if (self._config.llm_config.context_limit or 0) > 0
                        else None
                    ),
                )

                # Layer 1: Semantic Model Router
                router_tier = None
                if self._semantic_router is not None:
                    router_tier = self._semantic_router.route(req, context=self._context)

                # Layer 3: Execution Mode Gating & Planning
                intent = ExecutionIntent.DIRECT
                if router_tier == LLMTier.DEPTH_TIER:
                    intent = ExecutionIntent.PLANNING

                if intent == ExecutionIntent.PLANNING and self._plan_generator is not None:
                    plan = self._plan_generator.generate_plan(content_input, context=self._context)
                    if self._publisher is not None:
                        await self._publisher.publish(
                            AgentEvent(
                                type=EventType.PLAN_CREATED,
                                payload={
                                    "plan_id": plan.id,
                                    "steps": [s.model_dump() for s in plan.steps],
                                    # P6 in-band attribution travels with the value onto
                                    # the decision plane too: a subscriber must be able to
                                    # tell a heuristic plan from a model-authored one.
                                    "provenance": plan.provenance.model_dump(mode="json"),
                                },
                                session_id=self._context.session_id,
                            )
                        )

                # The agentic loop (P4, amended 2026-05: "agent steps are expected, not
                # rationed"). One model invocation and its tool round is an **agent step**;
                # the model must see what its tools returned, or it cannot answer from them.
                # Before this loop the turn was a single `generate` plus one tool round, and
                # returned — so a tool-using turn produced no answer at all, and the user was
                # left to ask "결과 안보임" about a search that had in fact run.
                #
                # `max_steps` bounds *this* loop, which is the only thing in the system that
                # can spin. It needs no `continuation` flag and no protocol change: the
                # counter is local to one externally-initiated request and therefore resets
                # with it by construction.
                # Calls the model wrote into `content` instead of into `tool_calls`. They
                # are counted and never executed: see `agent/text_tool_calls.py` and #694.
                # Without this the only trace of a discarded invocation is the JSON blob
                # sitting in the answer, which reads identically to a model that chose to
                # answer in prose.
                text_emitted: list[str] = []
                shown_tool_names = {t.name for t in tool_defs}
                assistant_msg_idx: int | None = None
                first_assistant_msg: ChatMessage | None = None
                # The answer a nudge rejected, while it waits for the retry's message to
                # take its place. See `_nudged_retry` for why it is not removed earlier.
                superseded: ChatMessage | None = None
                first_answer: str | None = None
                resp_content = ""
                tool_calls: tuple[ToolCallRequest, ...] = ()
                evidence_nudged = False
                # The second latch. `evidence_nudged` covers a turn that consulted nothing;
                # this one covers the turn that consulted *something* and answered past it,
                # which is the median shape in the #697 baseline -- one tool call against a
                # median declared horizon of six. Each reason fires at most once, both are
                # set before their retry and neither is cleared inside the turn, so the
                # extra cost of the whole mechanism is bounded at two steps regardless of
                # what the model does. That bound is what stops this becoming "the runtime
                # second-guesses the model indefinitely".
                #
                # The bound is **per turn, not per session** (review of #739). Both latches
                # are declared here, inside `execute_turn`, so a new turn gets new ones: a
                # model that is persistently ungrounded pays up to two extra steps on every
                # turn of the conversation, not two across all of them. That is correct --
                # each turn is a new question and has to be checked on its own evidence --
                # but "at most two extra steps" reads as a session-wide budget and is not
                # one, so anyone costing this mechanism should multiply by turns.
                grounding_nudged = False
                artifact_nudged = False
                unsupported: tuple[str, ...] = ()
                # The runtime's own nudges, kept so they can be subtracted from the support
                # set. The grounding nudge quotes the unsupported specifics back, so left in
                # it would ground them and the check would disarm itself one step after
                # firing.
                injected_nudges: set[str] = set()
                while True:
                    step = self._run_steps + 1
                    if step > self._config.max_steps:
                        # Terminate with an explicit envelope, never silently (P4/P6). The
                        # partial work stands in history; what is refused is another step.
                        self.transition_to(AgentState.ERROR)
                        budget_err = (
                            f"Agent step budget exceeded: maximum {self._config.max_steps} "
                            f"steps in a single request"
                        )
                        stop_reason = "step_budget_exceeded"
                        logger.warning("Step budget exceeded for agent %s", self.agent_id)
                        return TurnResult(
                            turn_index=self._turn_counter,
                            steps_taken=self._run_steps,
                            content=resp_content,
                            tool_calls=tuple(all_tool_calls),
                            tool_executions=tuple(tool_executions),
                            is_completed=False,
                            text_emitted_tool_calls=tuple(text_emitted),
                            error=budget_err,
                            stop_reason=stop_reason,
                            correlation_id=correlation_id,
                            provenance=None,
                            persona=self.persona_name or self.persona,
                            active_skills=live_active_skills,
                            # Re-read rather than using `loaded_skills_snapshot`: this
                            # refusal is returned *after* steps have run, so a skill the
                            # turn's own `load_skill` call brought in is already in
                            # session context and the pre-turn snapshot would omit it.
                            # Same reason the success return below re-reads.
                            loaded_skills=tuple(sorted(self._loaded_skills)),
                            usage=aggregate_token_usages(step_usages),
                        )

                    # The counter a reporting surface can read. `step` is a local and
                    # cannot outlive the return, so nothing outside sees the count unless
                    # it is mirrored here. Assigned rather than incremented, so it is
                    # cleared by the first step of every externally initiated request and
                    # never accumulates across a conversation — that accumulation is the
                    # defect this branch exists to remove.
                    #
                    # It is assigned *after* the ceiling check, so it counts steps taken
                    # rather than attempted: `run_steps` never exceeds `max_steps`, and
                    # `steps_remaining` bottoms out at exactly zero.
                    self._run_steps = step

                    durable_events.append(
                        {
                            "type": "ASSISTANT_ATTEMPT",
                            "step": step,
                            "turn_index": self._turn_counter,
                        }
                    )
                    durable_events.append(
                        {
                            "type": "REQUEST_CONTEXT",
                            **self._request_context_fields(step, req, req_layers),
                        }
                    )

                    await _emit_stream(
                        "status",
                        {"status": "generating", "detail": "Generating response..."},
                    )
                    resp = await self._invoke_step_model(
                        llm, req, step, stream_callback, durable_events, step_usages
                    )
                    resp_content = resp.content or ""
                    tool_calls = resp.tool_calls

                    if (
                        not tool_calls
                        and not resp_content
                        and resp.finish_reason == FinishReason.LENGTH
                    ):
                        raise TokenBudgetExhaustedError(
                            "Model token budget exhausted during reasoning before an answer could be produced"
                        )

                    all_tool_calls.extend(tool_calls)

                    # Only when the step produced no structured call. A step that did call
                    # a tool and also happened to print JSON is not the defect: the
                    # invocation channel worked, so nothing was discarded.
                    if not tool_calls:
                        text_emitted.extend(
                            detect_text_emitted_tool_calls(resp_content, shown_tool_names)
                        )

                    if tool_calls or resp_content:
                        # The retry wrote something, so it takes the rejected answer's
                        # place: never two `ASSISTANT` turns in a row. Removed only if it
                        # is still the last message.
                        if (
                            superseded is not None
                            and self._history
                            and self._history[-1] is superseded
                        ):
                            self._history.pop()
                        superseded = None
                        assistant_msg_idx = len(self._history)
                        self._history.append(
                            redact_message(
                                ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=resp_content or None,
                                    # This step's calls only. The message is the record of one
                                    # invocation; the turn's accumulation belongs on TurnResult.
                                    tool_calls=tool_calls,
                                )
                            )
                        )
                        if resp_content:
                            durable_events.append(
                                {
                                    "type": "ASSISTANT_MESSAGE",
                                    "content": resp_content,
                                }
                            )
                        if tool_calls:
                            for tc in tool_calls:
                                durable_events.append(
                                    {
                                        "type": "TOOL_CALL",
                                        "tool_call_id": tc.id,
                                        "name": tc.name,
                                        "arguments": tc.arguments,
                                    }
                                )

                    # No tool calls: the model has answered, and the step run ends --
                    # unless the answer rests on nothing and this agent was configured to
                    # refuse that. #697 measured the cost of accepting it: against problems
                    # declaring a median horizon of six dependent steps, the median turn
                    # took one, and twenty-eight of a hundred answered with no tool call at
                    # all. The loop was not failing to find things; it was not looking, and
                    # nothing compared the effort spent to what the task needed.
                    if not tool_calls or self._tools is None:
                        # Nothing ran, or nothing that ran found anything. The second case
                        # is the larger one: in the first live baseline 28 problems of 100
                        # made no tool call, and **38 stopped at exactly one** -- a search
                        # that missed, read as an absence, ending the turn (#698). The
                        # first case got a second chance and the bigger one did not.
                        nothing_found = not any(
                            _tool_outcome_of(record) == ToolOutcome.PRODUCTIVE.value
                            for record in tool_executions
                        )
                        # Deferred to the grounding nudge when the answer has a specific
                        # nothing supports: "the figure 100 appears in nothing you read" is
                        # strictly more useful than "your search found nothing", and only
                        # one of the two may fire per turn. So this covers the case
                        # grounding cannot -- an answer that asserts no specific at all,
                        # which is what "the line does not appear in the file" is, and what
                        # 38 of the baseline's 100 problems ended with.
                        defer_to_grounding = bool(tool_executions) and bool(
                            unsupported_specifics(
                                resp_content,
                                _grounding_supports(req.messages, tool_executions, injected_nudges),
                            )
                        )
                        if (
                            self._config.require_evidence_before_answer
                            and self._tools is not None
                            and nothing_found  # evidence nudge when tools found nothing
                            and not defer_to_grounding
                            and not evidence_nudged
                        ):
                            # Once. Without the latch this is an unbounded exchange with a
                            # model that answers in prose, ended only by the step budget --
                            # which converts a cheap wrong answer into an expensive one.
                            evidence_nudged = True
                            first_answer = resp_content
                            # This step's answer, or none. With no tool calls a message
                            # was appended this step exactly when there was content; an
                            # empty answer leaves `assistant_msg_idx` on an earlier step's
                            # tool-call message, which is not the answer being rejected.
                            rejected_idx: int | None = assistant_msg_idx if resp_content else None
                            first_assistant_msg = (
                                cast(ChatMessage, self._history[rejected_idx])
                                if rejected_idx is not None
                                else None
                            )
                            injected_nudges.add(EVIDENCE_REQUIRED_NUDGE)
                            durable_events.append(
                                {
                                    "type": "EVIDENCE_NUDGE",
                                    "step": step,
                                    "turn_index": self._turn_counter,
                                }
                            )
                            # In this request's turn context only, never in `_history`.
                            # The nudge is a runtime artifact, not something the user said:
                            # in history it persists, reaches `agent.history` and the CLI
                            # transcript, and accumulates one synthetic user turn per turn
                            # for the rest of the session (review of #702). A distinct
                            # durable event records that it happened, which is what the
                            # log needs; the conversation does not need a fake line in it.
                            # How the retry is built, and what happens to the answer it
                            # rejects, is `_nudged_retry` (#1420).
                            superseded = first_assistant_msg
                            req, req_layers = self._nudged_retry(
                                req,
                                EVIDENCE_REQUIRED_NUDGE,
                                turn_extra_sections,
                            )
                            continue

                        if (
                            evidence_nudged
                            # Not merely `is not None`: an empty first answer never
                            # entered history, so there is nothing to fall back to, and
                            # "declining" would discard the retry's answer for "" (#1420).
                            and first_answer
                            and _is_evidence_nudge_declined(resp_content, first_answer)
                        ):
                            # The model declined the evidence nudge by stating the question is
                            # answerable from what was given or providing a materially equivalent
                            # response (#756, #784). Accept the first answer and restore it into
                            # history in place of the retry's, so the turn still records one
                            # answer. The retry's message took the first one's place.
                            resp_content = first_answer
                            durable_events.append(
                                {
                                    "type": "EVIDENCE_NUDGE_DECLINED",
                                    "step": step,
                                    "turn_index": self._turn_counter,
                                }
                            )
                            if first_assistant_msg is not None:
                                if assistant_msg_idx is not None and assistant_msg_idx < len(
                                    self._history
                                ):
                                    self._history[assistant_msg_idx] = first_assistant_msg
                                else:
                                    assistant_msg_idx = len(self._history)
                                    self._history.append(first_assistant_msg)
                            elif assistant_msg_idx is not None and assistant_msg_idx < len(
                                self._history
                            ):
                                # The first answer was empty and so never entered history.
                                self._history.pop(assistant_msg_idx)
                                assistant_msg_idx = None

                        # What the answer asserts that nothing this turn read contains.
                        # Computed on every turn, acted on only when the agent asked for
                        # it: the observation is what #700 compares against a declared
                        # horizon and what a fabrication signal can be built from now that
                        # traps have left the eval set (#733), and gating the field on the
                        # setting would withhold it from every agent that has not opted in.
                        unsupported = unsupported_specifics(
                            resp_content,
                            _grounding_supports(req.messages, tool_executions, injected_nudges),
                        )
                        if (
                            self._config.require_evidence_before_answer
                            and self._tools is not None
                            and unsupported
                            and tool_executions
                            and not grounding_nudged
                        ):
                            # `tool_executions` non-empty is what keeps the two reasons
                            # disjoint at the point of firing: with no execution the
                            # evidence nudge above has already asked, and asking twice for
                            # the same silence spends a step to repeat a question.
                            grounding_nudged = True
                            grounding_nudge = (
                                f"{GROUNDING_REQUIRED_NUDGE_PREFIX}{describe(unsupported)}"
                                f"{GROUNDING_REQUIRED_NUDGE_SUFFIX}"
                            )
                            injected_nudges.add(grounding_nudge)
                            durable_events.append(
                                {
                                    "type": "GROUNDING_NUDGE",
                                    "step": step,
                                    "turn_index": self._turn_counter,
                                    "unsupported": list(unsupported),
                                }
                            )
                            # Request-scoped, never `_history`, for the reason #702 records
                            # for the evidence nudge: in history it persists into the
                            # session, the CLI transcript and every later turn.
                            superseded = (
                                self._history[assistant_msg_idx]
                                if resp_content and assistant_msg_idx is not None
                                else None
                            )
                            req, req_layers = self._nudged_retry(
                                req,
                                grounding_nudge,
                                turn_extra_sections,
                            )
                            continue

                        # Missing artifact image hallucination check (Tier 1 Nudge):
                        art_nudge_info = _evaluate_artifact_nudge(
                            resp_content, self._tools, self.workspace_root, artifact_nudged
                        )
                        if art_nudge_info is not None:
                            artifact_nudged = True
                            missing_path, artifact_nudge = art_nudge_info
                            injected_nudges.add(artifact_nudge)
                            durable_events.append(
                                {
                                    "type": "ARTIFACT_NUDGE",
                                    "step": step,
                                    "turn_index": self._turn_counter,
                                    "missing_artifact": missing_path,
                                }
                            )
                            msg_to_supersede = (
                                self._history[assistant_msg_idx]
                                if resp_content and assistant_msg_idx is not None
                                else None
                            )
                            superseded = msg_to_supersede
                            req, req_layers = self._nudged_retry(
                                req,
                                artifact_nudge,
                                turn_extra_sections,
                            )
                            continue

                        if evidence_nudged and grounding_nudged:
                            stop_reason = "model_stopped_after_both_nudges"
                        elif grounding_nudged:
                            stop_reason = "model_stopped_after_grounding_nudge"
                        elif evidence_nudged or artifact_nudged:
                            stop_reason = "model_stopped_after_nudge"
                        elif tool_executions and nothing_found:
                            stop_reason = "model_stopped_after_unproductive_tools"
                        else:
                            stop_reason = "model_stopped"
                        if self._tools is None:
                            stop_reason = "no_tools_registered"
                        break

                    self.transition_to(AgentState.CALLING_TOOL)
                    tools_unreported = True
                    tool_messages, step_executions = await self._execute_tools(
                        tool_calls, stream_callback=stream_callback
                    )
                    # Redacted, then held to the result cap (#1422): an over-cap result is
                    # stored in full and the history keeps an excerpt naming it. Decided
                    # here, once; later steps render the same excerpt.
                    reader_offered = any(d.name == TOOL_RESULT_READ_TOOL for d in tool_defs)
                    step_results = [redact_message(m) for m in tool_messages]
                    self._history.extend(
                        self._ingest_tool_message(m, readable=reader_offered) for m in step_results
                    )
                    tool_executions.extend(step_executions)
                    tools_unreported = False
                    for tr in step_executions:
                        durable_events.append(
                            {
                                "type": "TOOL_RESULT",
                                "tool_call_id": tr.tool_call_id,
                                "name": tr.tool_name,
                                "output": canonical_tool_text(tr.output),
                                # `status` alone cannot tell a search that answered from
                                # one that matched nothing -- both are "success" (#698).
                                # Without this the log shows a healthy call before a turn
                                # that gave up, and the giving up looks unmotivated.
                                "status": tr.status,
                                "outcome": _tool_outcome_of(tr),
                                "at": _now_iso(),
                                "duration_ms": tr.duration_ms,
                            }
                        )
                    self.transition_to(AgentState.REASONING)
                    await _emit_stream(
                        "status",
                        {"status": "thinking", "detail": "Processing tool results..."},
                    )

                    # Between steps is a boundary too (#1422): a step whose results push
                    # the next request over the threshold compacts now, before that
                    # request is built, instead of sending it over. Safe mid-turn because
                    # this step's calls and results are all in history, and compaction
                    # cuts only before a user turn. This step itself is held out of the
                    # pass: the model has not read its results yet.
                    if await self._auto_compact_if_needed(
                        tool_defs,
                        turn_extra_sections,
                        reason="auto_threshold_mid_turn",
                        hold_unseen_step=True,
                    ):
                        # Compaction replaces the history list; point at this step's
                        # assistant message in the new one.
                        assistant_msg_idx = next(
                            (
                                index
                                for index in range(len(self._history) - 1, -1, -1)
                                if self._history[index].role == MessageRole.ASSISTANT
                            ),
                            None,
                        )

                    # The step's own results, together, must still fit the window (#1480).
                    # Each is under the result cap, but several can exceed what the
                    # compacted history leaves; they are cut to excerpts of an equal share,
                    # or, when even that cannot fit, the step is refused and nothing over
                    # the window is sent.
                    step_refusal = self._fit_step_to_window(
                        step_results, tool_defs, turn_extra_sections, readable=reader_offered
                    )
                    if step_refusal is not None:
                        withheld_calls = self._withhold_refused_step(turn_live)
                        self.transition_to(AgentState.ERROR)
                        stop_reason = "step_results_over_window"
                        return TurnResult(
                            turn_index=self._turn_counter,
                            steps_taken=self._run_steps,
                            content="",
                            tool_calls=tuple(all_tool_calls),
                            tool_executions=tuple(tool_executions),
                            is_completed=False,
                            text_emitted_tool_calls=tuple(text_emitted),
                            error=step_refusal,
                            stop_reason=stop_reason,
                            correlation_id=correlation_id,
                            provenance=None,
                            persona=self.persona_name or self.persona,
                            active_skills=live_active_skills,
                            loaded_skills=tuple(sorted(self._loaded_skills)),
                        )

                    # Rebuild from the history that now holds the tool results. This is the
                    # line the whole amendment is about: without it the model never sees
                    # what its own tools returned.
                    req_layers = self._prepare_turn_layers(extra_sections=turn_extra_sections)
                    step_messages = assemble_request_messages(req_layers)
                    req = req.model_copy(update={"messages": tuple(step_messages)})

                self.transition_to(AgentState.EMITTING_RESPONSE)
                await _emit_stream(
                    "status",
                    {"status": "generating", "detail": "Finalizing response..."},
                )

                # Execute POST_TURN hook
                post_turn_ctx = HookContext(
                    agent_id=self.agent_id,
                    session_id=self._context.session_id,
                    trace_id=self._tracer.trace_id,
                    event_type=HookEvent.POST_TURN,
                    payload={
                        "content": resp_content,
                        "turn_index": self._turn_counter,
                        "tool_calls_count": len(tool_calls),
                        "tool_executions_count": len(tool_executions),
                    },
                )
                post_turn_decision = await self._hook_runner.run_hooks(
                    HookEvent.POST_TURN, post_turn_ctx
                )
                if (
                    post_turn_decision.action == HookAction.MODIFY
                    and post_turn_decision.modified_payload
                ):
                    if "content" in post_turn_decision.modified_payload:
                        resp_content = str(post_turn_decision.modified_payload["content"])
                        if assistant_msg_idx is not None and assistant_msg_idx < len(self._history):
                            orig_msg = self._history[assistant_msg_idx]
                            self._history[assistant_msg_idx] = redact_message(
                                ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=resp_content or None,
                                    tool_calls=orig_msg.tool_calls,
                                )
                            )
                        elif resp_content:
                            assistant_msg_idx = len(self._history)
                            self._history.append(
                                redact_message(
                                    ChatMessage(
                                        role=MessageRole.ASSISTANT,
                                        content=resp_content,
                                    )
                                )
                            )

                resp_content = _apply_artifact_sanitization(
                    resp_content, self.workspace_root, self._history, assistant_msg_idx
                )

                # P6: the connector's attribution is propagated verbatim, including
                # `None`. It is NOT defaulted to a synthetic "agent.core/BaseAgent
                # primary" — that would name this component as the server of a value
                # an LLM produced, turning "not stated" into a positively asserted
                # clean primary result. `core/provenance.py`: "a missing marker read
                # as 'nothing went wrong' is the default-masquerading-as-a-real-answer
                # that Principle 6 exists to forbid." An unattributed response is
                # carried as `None` and rejected by the consumer that needs it.
                # A field nobody reads is not a remedy. `text_emitted_tool_calls` travels on
                # the result and into the eval report, and neither surface is in front of
                # the person actually holding a model whose calls fall on the floor: they
                # see a JSON fragment where an answer should be and nothing that says why.
                # Gated on no tool having executed, so a turn that used the structured
                # channel and merely also printed JSON stays quiet -- nothing was lost
                # there. The detector is conservative but not exact (a quoted tool schema
                # is call-shaped), which is why this is a warning naming the remedy rather
                # than an error: the cost of a false one is a log line.
                if text_emitted and not tool_executions:
                    logger.warning(
                        "Agent %s: model %r wrote %d tool call(s) into message content "
                        "instead of the structured tool_calls field, so none ran and the "
                        "answer is the discarded call itself (#694): %s. Remedy at the "
                        "model, not here: use one whose calls arrive structured "
                        "(qwen3:8b, qwen2.5:7b-instruct are verified), or `ollama create` "
                        "a variant whose template the model honours.",
                        self.agent_id,
                        self._config.llm_config.model_name,
                        len(text_emitted),
                        ", ".join(sorted(set(text_emitted))),
                    )

                self._active_session.updated_at = _now_iso()
                self.transition_to(AgentState.IDLE)
                return TurnResult(
                    turn_index=self._turn_counter,
                    steps_taken=self._run_steps,
                    content=resp_content,
                    tool_calls=tuple(all_tool_calls),
                    tool_executions=tuple(tool_executions),
                    is_completed=True,
                    unsupported_claims=tuple(unsupported),
                    text_emitted_tool_calls=tuple(text_emitted),
                    stop_reason=stop_reason,
                    correlation_id=correlation_id,
                    provenance=resp.provenance,
                    router_tier=router_tier.value if router_tier else None,
                    persona=self.persona_name or self.persona,
                    active_skills=live_active_skills,
                    loaded_skills=tuple(sorted(self._loaded_skills)),
                    usage=aggregate_token_usages(step_usages),
                )
            except (BudgetExceededError, TokenBudgetExhaustedError) as exc:
                # P6, "Error classification never authorises substitution", classification
                # table: "| Quota or budget ceiling exceeded | No — must propagate | No |".
                # A ceiling breach is not retry- or failover-eligible, so it must not be
                # dressed as one. The generic handler below returns `path=FAILOVER` with
                # `served_by=agent.core/error_handler`, an `attempts` record, a
                # `failover.event` span and a `PROVIDER_FAILOVER` bus notice — describing a
                # provider substitution attempt that never happened. That is the silent
                # substitution P6 forbids, running in the reporting direction: a refusal
                # reported as a degraded answer from a substitute.
                #
                # The refusal propagates as itself, in the same envelope shape the step
                # budget refusal above already uses: `is_completed=False`, the ceiling's own
                # message, and `provenance=None` because no path produced a value.
                self.transition_to(AgentState.ERROR)
                if isinstance(exc, BudgetExceededError):
                    # Stated, not left to the wording of `error` (#969): the ledger only
                    # grows, so a retry meets this ceiling again until the ceiling changes,
                    # and a head deciding whether to offer Retry reads it from here.
                    #
                    # This also matches `StepBudgetExceededError`, which subclasses it.
                    # Nothing in `src/` raises that class today, and the step ceiling
                    # returns its own `step_budget_exceeded` result before reaching this
                    # handler, so no behaviour depends on the overlap. Whoever raises it
                    # first should decide whether a step ceiling is a refusal a retry
                    # meets again -- it is not, since `run_steps` resets per turn -- and
                    # narrow this test rather than inherit the label by accident.
                    stop_reason = "budget_exceeded"
                logger.warning(
                    "Budget ceiling reached or token budget exhausted for agent %s: %s",
                    self.agent_id,
                    exc,
                )
                budget_err_ctx = HookContext(
                    agent_id=self.agent_id,
                    session_id=self._context.session_id,
                    trace_id=self._tracer.trace_id,
                    event_type=HookEvent.ON_ERROR,
                    payload={
                        "error": str(exc),
                        "error_class": type(exc).__name__,
                        "turn_index": self._turn_counter,
                    },
                )
                try:
                    await self._hook_runner.run_hooks(HookEvent.ON_ERROR, budget_err_ctx)
                except Exception:
                    logger.debug("Error running ON_ERROR hook", exc_info=True)
                return TurnResult(
                    turn_index=self._turn_counter,
                    steps_taken=self._run_steps,
                    content="",
                    # What ran before the failure, as the step-budget refusal reports it.
                    # `()` here claimed an errored turn used no tools (#1366).
                    tool_calls=tuple(all_tool_calls),
                    tool_executions=tuple(tool_executions),  # ran before the ceiling
                    tool_executions_complete=not tools_unreported,  # a ceiling mid-step
                    is_completed=False,
                    error=str(exc),
                    stop_reason=stop_reason,
                    correlation_id=correlation_id,
                    provenance=None,
                    persona=self.persona_name or self.persona,
                    active_skills=live_active_skills,
                    # Re-read for the same reason as the step-budget refusal above: the
                    # ceiling can be reached on any step's pre-flight check, so skills
                    # loaded earlier in this turn belong in the envelope.
                    loaded_skills=tuple(sorted(self._loaded_skills)),
                    usage=aggregate_token_usages(step_usages),
                )
            except (Exception, asyncio.CancelledError) as exc:
                if isinstance(exc, asyncio.CancelledError):
                    stop_reason = "cancelled"
                    logger.info("Turn %d cancelled for agent %s", self._turn_counter, self.agent_id)
                    if (
                        self._state is not AgentState.IDLE
                        and self._state is not AgentState.TERMINATED
                    ):
                        self.transition_to(AgentState.IDLE)
                    raise
                self.transition_to(AgentState.ERROR)
                # Named before anything else reads it. `error` carries the same fact as
                # prose, and prose is not something a caller can branch on without
                # guessing at a provider's wording (#1277).
                stop_reason, failure = _turn_failure(exc, stop_reason, self.agent_id)
                req_model = self._config.llm_config.model_name

                # Execute ON_ERROR hook
                err_ctx = HookContext(
                    agent_id=self.agent_id,
                    session_id=self._context.session_id,
                    trace_id=self._tracer.trace_id,
                    event_type=HookEvent.ON_ERROR,
                    payload={
                        "error": str(exc),
                        "error_class": type(exc).__name__,
                        "turn_index": self._turn_counter,
                    },
                )
                try:
                    await self._hook_runner.run_hooks(HookEvent.ON_ERROR, err_ctx)
                except Exception:
                    logger.debug("Error running ON_ERROR hook", exc_info=True)

                # P6 Check 5 (#148, #175, #180, #202): emit a `failover.event` span in telemetry and capture span_id
                span_id = self._tracer.start_span(
                    FAILOVER_EVENT_SPAN_NAME,
                    attributes={
                        "requested_provider": self.agent_id,
                        "served_provider": "agent.core",
                        "served_model": "error_handler",
                        "error_class": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                self._tracer.end_span(
                    span_id=span_id,
                    status=SpanStatus.ERROR,
                    error_message=str(exc),
                )

                failover_provenance = Provenance(
                    path=ExecutionPath.FAILOVER,
                    requested=ServiceRef(
                        provider=self.agent_id,
                        model=req_model,
                    ),
                    served_by=ServiceRef(
                        provider="agent.core",
                        model="error_handler",
                    ),
                    attempts=(
                        AttemptRecord(
                            provider=self.agent_id,
                            model=req_model,
                            error_class=type(exc).__name__,
                            span_id=span_id,
                        ),
                    ),
                )

                # P6 Check 4 (#148, #175, #180): publish PROVIDER_FAILOVER notice strictly ordered BEFORE reply/return
                if self._bus is not None:
                    failover_payload: dict[str, Any] = {
                        "requested_provider": self.agent_id,
                        "served_provider": "agent.core",
                        "error_class": type(exc).__name__,
                        "error_message": str(exc),
                        "session_id": self._context.session_id,
                    }
                    failover_payload["span_id"] = span_id
                    if req_model:
                        failover_payload["requested_model"] = req_model

                    failover_event = AgentEvent(
                        type=EventType.PROVIDER_FAILOVER,
                        recipient_id=self.agent_id,
                        topic="agent.chat.failover",
                        priority=EventPriority.NORMAL,
                        payload=failover_payload,
                        provenance=failover_provenance,
                        trace_id=self._tracer.trace_id,
                    )
                    if self._publisher is not None:
                        await self._publisher.publish(failover_event)
                    else:
                        await self._bus.publish(failover_event)

                return TurnResult(
                    turn_index=self._turn_counter,
                    steps_taken=self._run_steps,
                    content="",
                    # What ran before the failure, as the step-budget refusal reports it.
                    # `()` here claimed an errored turn used no tools (#1366).
                    tool_calls=tuple(all_tool_calls),
                    tool_executions=tuple(tool_executions),  # ran before the failure
                    tool_executions_complete=not tools_unreported,  # a failure mid-step
                    is_completed=False,
                    error=failure,
                    stop_reason=stop_reason,
                    correlation_id=correlation_id,
                    provenance=failover_provenance,
                    persona=self.persona_name or self.persona,
                    active_skills=live_active_skills,
                    loaded_skills=loaded_skills_snapshot,
                    usage=aggregate_token_usages(step_usages),
                )
            finally:
                if stop_reason == "cancelled":
                    outcome = "cancelled"
                else:
                    outcome = "completed" if self._state == AgentState.IDLE else "error"
                durable_events.append(
                    {
                        "type": "TURN_END",
                        "turn_index": self._turn_counter,
                        "outcome": outcome,
                        # The three figures a step-run post-mortem starts from. Without
                        # them the log says a turn happened and not what it did.
                        "steps": self._run_steps,
                        "tool_executions": len(tool_executions),
                        "stop_reason": stop_reason,
                        "at": _now_iso(),
                    }
                )
                # The floor under every way a turn can stop (#1423): a step whose tools
                # were asked for and not all answered is dropped, never persisted. Not
                # closed with invented "cancelled" results -- a tool may have run and
                # changed something, and a result no tool produced would say otherwise
                # (P6). The dropped message was never sent as input to any request, so
                # removing it rewrites nothing a model saw, and history stays append-only
                # for every request that was sent. The log keeps the calls as `TOOL_CALL`
                # events and this drop beside them.
                #
                # A turn that completed is the exception (#1495). It can end on a message
                # that asks for tools only when no tool could answer (no registry), and
                # that message's text is then the turn's answer: the room lands it on the
                # transcript. Dropping it made the model forget what the room remembers,
                # so only the calls are stripped and the text stays. A completed message
                # with no text has nothing to keep, and goes as before.
                turn_live = self._live_session(turn_session_id)
                turn_live.last_turn_tool_calls = all_tool_calls[
                    : len(all_tool_calls) - withheld_calls
                ]
                turn_messages = turn_live.messages
                dangling = _unanswered_tool_step(turn_messages)
                kept_text = False
                if dangling is not None:
                    dangling_index, unanswered_ids = dangling
                    asked = turn_messages[dangling_index]
                    kept_text = outcome == "completed" and bool(asked.content)
                    if kept_text:
                        dropped_count = len(turn_messages) - dangling_index - 1
                        turn_messages[dangling_index] = asked.model_copy(update={"tool_calls": ()})
                        del turn_messages[dangling_index + 1 :]
                    else:
                        dropped_count = len(turn_messages) - dangling_index
                        del turn_messages[dangling_index:]
                    logger.warning(
                        "Dropped an unanswered tool step (%d message(s), calls %s, text "
                        "kept: %s) from session %s of agent %s after the turn ended %s",
                        dropped_count,
                        ", ".join(unanswered_ids),
                        kept_text,
                        turn_session_id,
                        self.agent_id,
                        outcome,
                    )
                    durable_events.append(
                        {
                            "type": "TOOL_STEP_DROPPED",
                            "turn_index": self._turn_counter,
                            "outcome": outcome,
                            "unanswered_tool_call_ids": list(unanswered_ids),
                            "dropped_message_count": dropped_count,
                            "assistant_text_kept": kept_text,
                        }
                    )
                # Stamped here rather than at each of the six append sites: one place to
                # read, and no way for a seventh append to be added without it. The queue
                # is agent-wide, so without this an entry cannot be told from another
                # session's — which is what made `reset_session` a choice between deleting
                # other sessions' events and keeping the wrong ones (#670).
                for event in durable_events:
                    event["session_id"] = turn_session_id
                self._pending_durable_events.extend(durable_events)

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

    def _refused_tool_call(
        self,
        tc: ToolCallRequest,
        err_msg: str,
        duration_ms: float,
    ) -> tuple[ChatMessage, ToolExecutionRecord]:
        """The synthesized tool message and record for a call the pre-tool path refused.

        One function rather than two identical literals, because the duplication carried
        a defect. `ToolExecutionRecord.arguments` is `ImmutableJsonMapping`, and
        `tc.arguments` is frozen *recursively*, so a shallow `dict()` copy of it left
        nested `MappingProxyType`s in place. Those do not survive the field's
        `JsonValue` validation — `AfterValidator(freeze_mapping)` runs after it, so the
        field rejects the shallow copy rather than re-freezing it — and the refusal path
        raised `ValidationError` instead of returning the refusal it exists to return
        (#665).
        """
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content=err_msg,
            name=tc.name,
            tool_call_id=tc.id,
        )
        rec = ToolExecutionRecord(
            tool_name=tc.name,
            arguments=cast(dict[str, Any], unwrap_immutable(tc.arguments)),
            output=None,
            status=ToolResultStatus.ERROR,
            error=err_msg,
            duration_ms=duration_ms,
            tool_call_id=tc.id,
        )
        return msg, rec

    async def _execute_single_tool(
        self,
        tc: ToolCallRequest,
        tool_ctx: ToolContext,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> tuple[ChatMessage, ToolExecutionRecord]:
        """Execute a single tool call within the provided ToolContext, intercepted by hooks."""
        t_start = asyncio.get_running_loop().time()

        # 1. Execute PRE_TOOL_USE hook
        pre_payload: dict[str, Any] = {
            "tool_name": tc.name,
            "tool_call_id": tc.id,
            "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
        }
        # What the tool declares about itself, so a hook can decide from that and not from
        # the tool's name (#1463). Absent for a name this agent cannot resolve.
        pre_tool = self._resolve_tool(tc.name)
        if pre_tool is not None:
            pre_payload["writes_files"] = tool_writes_files(pre_tool)
            pre_payload["spawns_subagents"] = tool_spawns_subagents(pre_tool)
        pre_ctx = HookContext(
            agent_id=self.agent_id,
            session_id=self._context.session_id,
            trace_id=self._context.trace_id,
            event_type=HookEvent.PRE_TOOL_USE,
            payload=pre_payload,
        )
        pre_decision = await self._hook_runner.run_hooks(HookEvent.PRE_TOOL_USE, pre_ctx)

        if pre_decision.action == HookAction.ASK:
            request_id = f"appr_{uuid.uuid4().hex[:8]}"

            sub = None
            if self._bus is not None:
                sub = self._bus.subscribe({f"session.{self._context.session_id}"})

            approval_arguments = cast(dict[str, Any], unwrap_immutable(tc.arguments))
            if self._bus is not None:
                req_evt = AgentEvent(
                    type=EventType.TOOL_APPROVAL_REQUEST,
                    topic=f"session.{self._context.session_id}",
                    sender_id=self.agent_id,
                    payload={
                        "request_id": request_id,
                        "tool_call_id": tc.id,
                        "tool_name": tc.name,
                        "arguments": approval_arguments,
                        "reason": pre_decision.reason,
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                    },
                    trace_id=self._context.trace_id,
                )
                if self._publisher is not None:
                    await self._publisher.publish(req_evt)
                else:
                    await self._bus.publish(req_evt)

            try:
                if sub is None:
                    raise TimeoutError()

                async def wait_for_response() -> Mapping[str, Any]:
                    while True:
                        evt = await sub.get()
                        if evt.type == EventType.TOOL_APPROVAL_RESPONSE:
                            payload = evt.payload
                            rid = payload.get("request_id")
                            tid = payload.get("tool_call_id")
                            if rid == request_id or tid == tc.id:
                                return payload

                payload = await asyncio.wait_for(
                    wait_for_response(),
                    timeout=getattr(self._config, "approval_timeout_seconds", 30.0),
                )
                from uclone_x.agent.hooks.models import ApprovalDecision

                # `payload` is `AgentEvent.payload` — frozen recursively, so
                # `payload.get("modified_arguments")` yields a `MappingProxyType`, while
                # `ApprovalDecision.modified_arguments` is `dict[str, Any]` under
                # `strict=True` and rejects it with `dict_type` (#665, #673). This
                # `ApprovalDecision(...)` sits in a `try` whose only handler is
                # `except TimeoutError`, so a MODIFY approval response would take the
                # turn down rather than degrade. Latent — no in-repo publisher sets the
                # key — and invisible to the fitness sweep, which matches
                # `dict(<expr>.<field>)` and not an aliased `.get()`.
                modified_arguments = payload.get("modified_arguments")
                decision = ApprovalDecision(
                    action=HookAction(payload.get("action", "allow")),
                    reason=payload.get("reason"),
                    modified_arguments=cast(dict[str, Any], unwrap_immutable(modified_arguments))
                    if modified_arguments is not None
                    else None,
                    decided_by=payload.get("decided_by"),
                )

                # Override pre_decision with human decision
                from uclone_x.agent.hooks.models import HookDecision

                pre_decision = HookDecision(
                    action=decision.action,
                    reason=decision.reason,
                    modified_payload={"arguments": decision.modified_arguments}
                    if decision.modified_arguments
                    else None,
                )
            except TimeoutError:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                return self._refused_tool_call(
                    tc, "Approval request timed out (denied fail-closed)", duration_ms
                )
            finally:
                if sub is not None:
                    sub.close()

        if pre_decision.action == HookAction.BLOCK:
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            block_reason = pre_decision.reason or "Blocked by hook"
            return self._refused_tool_call(
                tc, f"Tool execution blocked by hook: {block_reason}", duration_ms
            )

        effective_tc = tc
        if pre_decision.action == HookAction.MODIFY and pre_decision.modified_payload is not None:
            mod_payload = pre_decision.modified_payload
            if "arguments" in mod_payload and isinstance(mod_payload["arguments"], dict):
                effective_tc = tc.model_copy(update={"arguments": mod_payload["arguments"]})
            elif "tool_name" not in mod_payload and "arguments" not in mod_payload:
                effective_tc = tc.model_copy(update={"arguments": mod_payload})

        unwrapped_args: dict[str, Any] = cast(
            dict[str, Any], unwrap_immutable(effective_tc.arguments)
        )

        if stream_callback is not None:
            try:
                res_status = stream_callback(
                    "status",
                    {"status": "calling_tool", "detail": f"Running tool: {effective_tc.name}..."},
                )
                if asyncio.iscoroutine(res_status):
                    await res_status
                res_tool = stream_callback(
                    "tool_call",
                    {
                        "tool": effective_tc.name,
                        "args": unwrapped_args,
                        "output": None,
                        "status": "running",
                    },
                )
                if asyncio.iscoroutine(res_tool):
                    await res_tool
            except Exception:
                logger.debug("Error in stream_callback during tool start", exc_info=True)

        # The allowlist is checked here, not only at advertise time. `execute_turn` shows the
        # model the permitted names and this used to be a bare registry lookup, so a call the
        # model produced for a name it was never shown ran anyway -- and small models produce
        # exactly that, which is why `text_tool_calls` exists. Advertising is a hint; this is
        # the enforcement.
        #
        # Checked *before* the lookup so the two answers stay separate: a permitted-but-absent
        # tool and a present-but-forbidden one must not be reported with the same words.
        # `execute_tool_call` raises `PermissionError` and `KeyError` for the same distinction,
        # and a turn should not collapse what a direct call keeps apart. This check is the
        # *only* refusal: a `ScopedToolRegistry`'s lookup finds a registered tool outside its
        # scope (#908) -- the scope governs what `list_tools` advertises, not what `get` finds.
        allowed_names = self._config.allowed_tools
        if allowed_names and effective_tc.name not in allowed_names:
            err_msg = (
                f"Tool '{effective_tc.name}' is not in agent "
                f"'{self.agent_id}' allowed_tools {tuple(allowed_names)!r}"
            )
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            return (
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=err_msg,
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                ),
                ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=err_msg,
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                ),
            )

        tool_inst = self._resolve_tool(effective_tc.name)
        if tool_inst is None:
            err_msg = f"Tool '{effective_tc.name}' not found"
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=err_msg,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = ToolExecutionRecord(
                tool_name=effective_tc.name,
                arguments=unwrapped_args,
                output=None,
                status=ToolResultStatus.ERROR,
                error=err_msg,
                duration_ms=duration_ms,
                tool_call_id=effective_tc.id,
            )
        elif (refusal := self._capability_refusal(tool_inst)) is not None:
            # The persona flags, refused on the same path as `allowed_tools` above and in
            # the same shape (#1167): an error record, and the tool is never executed. After
            # the lookup rather than before it, because what the flags refuse is a kind of
            # tool, and only the instance says which kind it is.
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=refusal,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = ToolExecutionRecord(
                tool_name=effective_tc.name,
                arguments=unwrapped_args,
                output=None,
                status=ToolResultStatus.ERROR,
                error=refusal,
                duration_ms=duration_ms,
                tool_call_id=effective_tc.id,
            )
        else:
            # The tool's declarations, read once: a call that raises partway reports
            # them too, since failing does not undo a write (#1366).
            declared_writes = tool_writes_files(tool_inst)
            declared_spawns = tool_spawns_subagents(tool_inst)
            try:
                res = await tool_inst.execute(unwrapped_args, tool_ctx)
                duration_ms = (
                    res.execution_time_ms
                    if res.execution_time_ms > 0
                    else (asyncio.get_running_loop().time() - t_start) * 1000.0
                )

                # Update session plan natively if the tool was the plan tool
                if effective_tc.name == "update_plan" and res.success:
                    if isinstance(res.output, dict):
                        try:
                            action = str(res.output.get("action"))
                            if action == "create":
                                title = str(res.output.get("title", ""))
                                steps_val = res.output.get("steps")
                                steps_list = steps_val if isinstance(steps_val, list) else []
                                self.create_plan(
                                    title=title,
                                    steps=steps_list,  # type: ignore[reportArgumentType]
                                )
                            elif action == "update":
                                if not self.current_plan:
                                    raise RuntimeError("No active plan exists to update")
                                steps_val = res.output.get("steps")
                                if isinstance(steps_val, list):
                                    for step in steps_val:
                                        if isinstance(step, dict):
                                            idx = step.get("index")
                                            if isinstance(idx, int):
                                                verif = step.get("verification")
                                                self.update_step_status(
                                                    index=idx,
                                                    completed=bool(step.get("completed", False)),
                                                    verification=str(verif)
                                                    if verif is not None
                                                    else None,
                                                )
                                status_val = res.output.get("status")
                                if isinstance(status_val, str) and status_val:
                                    new_plan = self.current_plan.model_copy(
                                        update={"status": status_val}
                                    )
                                    self._live_session(self._context.session_id).plan = new_plan
                                    self._publish_plan_update(new_plan)

                            # Provide the new plan back to the tool result so LLM sees success explicitly
                            if self.current_plan:
                                plan_dump = {
                                    "plan_id": self.current_plan.plan_id,
                                    "title": self.current_plan.title,
                                    "status": self.current_plan.status,
                                    "steps": [s.model_dump() for s in self.current_plan.steps],
                                }
                            else:
                                plan_dump = {}
                            res = res.model_copy(
                                update={
                                    "output": {
                                        "message": f"Plan {action}d successfully",
                                        "plan": plan_dump,
                                    }
                                }
                            )
                        except Exception as e:
                            logger.error("Failed to update session plan from tool output: %s", e)
                            res = res.model_copy(
                                update={"success": False, "error": str(e), "output": None}
                            )

                # Canonical text, not `str()`: a dict result was a Python repr -- not JSON,
                # quoted with `'`, and ordered by insertion (#1422).
                content = canonical_tool_text(res.output) if res.success else str(res.error)
                # A call that succeeded and found nothing arrives structurally
                # indistinguishable from one that answered the question -- both are
                # `success=True`, and `{"total_matches": 0, "matches": []}` reads as an
                # absence. Measured: a model missed with a regex, concluded "the line does
                # not appear in the file", and stopped. The number was there (#698). The
                # note says what the result is; it does not say what to do next, which is
                # the model's judgement and not something the runtime can make for it.
                if classify_tool_outcome(res) is ToolOutcome.EMPTY:
                    content = f"{content}\n{EMPTY_RESULT_NOTE}"
                if (
                    not res.success
                    and res.error
                    and ("Path traversal" in res.error or "PathTraversalError" in res.error)
                ):
                    logger.warning(
                        "Path traversal violation during tool '%s' execution: %s",
                        effective_tc.name,
                        res.error,
                        extra={
                            "agent_id": self.agent_id,
                            "session_id": self._context.session_id,
                            "tool_name": effective_tc.name,
                            "error": str(res.error),
                        },
                    )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=content,
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=res.output,
                    status=ToolResultStatus.SUCCESS if res.success else ToolResultStatus.ERROR,
                    error=res.error,
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # The tool's declarations, carried to whoever reads the turn: a room
                    # records a written file only from a tool that says it writes (#1354).
                    writes_files=tool_writes_files(tool_inst),
                    spawns_subagents=tool_spawns_subagents(tool_inst),
                )
            except PathTraversalError as exc:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                logger.warning(
                    "Path traversal violation during tool '%s' execution: %s",
                    effective_tc.name,
                    exc,
                    extra={
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                        "tool_name": effective_tc.name,
                        "error": str(exc),
                    },
                )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Path traversal violation: {exc}",
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=f"Path traversal violation: {exc}",
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # The tool ran and raised: it may have written before it did (#1366).
                    writes_files=declared_writes,  # refused partway
                    spawns_subagents=declared_spawns,  # refused partway
                )
            except Exception as exc:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                logger.warning(
                    "Tool '%s' execution failed with unexpected exception: %s",
                    effective_tc.name,
                    exc,
                    extra={
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                        "tool_name": effective_tc.name,
                        "error": str(exc),
                    },
                )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Tool execution failed: {type(exc).__name__}: {exc}",
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # As above: a tool that raised had already started (#1366).
                    writes_files=declared_writes,  # raised partway
                    spawns_subagents=declared_spawns,  # raised partway
                )

        # 2. Execute POST_TOOL_USE hook
        post_ctx = HookContext(
            agent_id=self.agent_id,
            session_id=self._context.session_id,
            trace_id=self._context.trace_id,
            event_type=HookEvent.POST_TOOL_USE,
            payload={
                "tool_name": effective_tc.name,
                "tool_call_id": effective_tc.id,
                "arguments": unwrapped_args,
                "output": rec.output,
                "status": rec.status,
                "error": rec.error,
                "duration_ms": rec.duration_ms,
            },
        )
        post_decision = await self._hook_runner.run_hooks(HookEvent.POST_TOOL_USE, post_ctx)
        if post_decision.action == HookAction.BLOCK:
            block_reason = post_decision.reason or "Blocked by post-tool hook"
            err_msg = f"Tool execution blocked by hook: {block_reason}"
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=err_msg,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = rec.model_copy(update={"status": "error", "error": err_msg, "output": None})
        elif (
            post_decision.action == HookAction.MODIFY and post_decision.modified_payload is not None
        ):
            mod_payload = post_decision.modified_payload
            if "output" in mod_payload:
                new_output = mod_payload["output"]
                rec = rec.model_copy(update={"output": new_output})
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=canonical_tool_text(new_output),
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )

        if stream_callback is not None:
            try:
                out_val = unwrap_immutable(
                    rec.output if rec.status == ToolResultStatus.SUCCESS else rec.error
                )
                res_tool_done = stream_callback(
                    "tool_call",
                    {
                        "tool": effective_tc.name,
                        "args": unwrapped_args,
                        "output": out_val,
                        "status": "completed",
                        "error": rec.error,
                        "duration_ms": rec.duration_ms,
                    },
                )
                if asyncio.iscoroutine(res_tool_done):
                    await res_tool_done
            except Exception:
                logger.debug("Error in stream_callback during tool finish", exc_info=True)

        return msg, rec

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
        )
        if content == msg.content:
            return msg
        return msg.model_copy(update={"content": content})

    def _window_model(self) -> str | None:
        """The model the next request is sent to, as the window is looked up for it."""
        named = _named_model(self._config.llm_config.model_name) or _named_model(
            getattr(self._llm, "model", None)
        )
        if named is not None:
            return named
        # A local server's window is per loaded model, so the model the connector resolves
        # for a request naming none is the one to ask about (#1447's `_default_model`).
        # Hosted providers keep the table lookup they had, keyed by the configured name.
        if getattr(self._llm, "provider_name", None) in SERVED_WINDOW_PROVIDERS:
            return _named_model(getattr(self._llm, "_default_model", None))
        return None

    def _context_window(self) -> int | None:
        """The window the compaction trigger and the step budget count against (#1372).

        The one resolution all three limit sites use -- `_should_compact_session`,
        `_request_over_threshold` and `_fit_step_to_window` -- so they cannot disagree.
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

        `step_results` are the step's results before the cap, in the order appended; the
        history's trailing tool-call group holds them as ingested. The budget is the
        window less the room kept for the reply (`_reply_reserve`, #1509) and everything
        else the next request sends -- the system turn, the tool schemas, the turn
        context and the already-compacted history. When the step's results exceed it,
        `step_result_caps` shares it out and each over-share result is ingested again at
        its share: an excerpt whose full body stays readable with `tool_result_read`.
        Nothing a request has shown is rewritten: the model has not seen this step yet.

        `None` when the next request fits. Otherwise the plain refusal the turn ends
        with: `STEP_NO_ROOM_MESSAGE` when the request leaves no room for the step at
        all, so the conversation and not the tools is the cause (#1509), and
        `STEP_OVER_WINDOW_MESSAGE` when the step cannot fit even as excerpts or the
        fitted request is still over. Without a known window there is nothing to fit
        against, and the per-result cap is the only bound.
        """
        window = self._context_window()
        if window is None:
            return None
        reserve = self._reply_reserve()

        def request_tokens() -> int:
            messages = tuple(self._prepare_turn_messages(extra_sections=extra_sections))
            return estimate_request_tokens(LLMRequest(messages=messages, tools=tuple(tools)))

        total = request_tokens()
        if total + reserve <= window:
            return None
        history = self._history
        start = unseen_step_start(history)
        tail = list(range(start + 1, len(history))) if start is not None else []
        if (
            start is None
            or len(tail) != len(step_results)
            or any(
                history[i].tool_call_id != m.tool_call_id
                for i, m in zip(tail, step_results, strict=False)
            )
        ):
            return STEP_OVER_WINDOW_MESSAGE
        # Everything but the step itself -- its calls and its results -- already reaches
        # the window: no share of any size fits, and fewer calls would not help.
        outside = total - estimate_message_tokens(history[start:])
        if outside + reserve >= window:
            logger.warning(
                "No room is left in a %d-token window for a step's %d tool results: the "
                "request without the step is %d tokens and %d are kept for the reply",
                window,
                len(tail),
                outside,
                reserve,
            )
            return STEP_NO_ROOM_MESSAGE
        current = [history[i].content or "" for i in tail]
        rest = total - sum(estimate_text_tokens(c) for c in current if c)
        # Each message's estimate rounds up, so one token apiece is kept back, and two
        # more for a turn-context block merged into the last result.
        budget = window - reserve - rest - len(tail) - 2
        caps = step_result_caps([len(c.encode("utf-8")) for c in current], budget * 4)
        if caps is None:
            logger.warning(
                "A step's %d tool results cannot fit a %d-token window even as excerpts; "
                "%d tokens are left for them",
                len(tail),
                window,
                budget,
            )
            return STEP_OVER_WINDOW_MESSAGE
        workspace = self._resolve_workspace_root()
        for index, raw, content, cap in zip(tail, step_results, current, caps, strict=True):
            if raw.content is None or len(content.encode("utf-8")) <= cap:
                continue
            shared = ingest_tool_text(
                raw.content,
                artifacts_dir=artifacts_dir_for(workspace) if workspace is not None else None,
                session_id=self._context.session_id,
                readable=readable,
                cap_bytes=cap,
            )
            history[index] = history[index].model_copy(update={"content": shared})
        fitted = request_tokens()
        if fitted + reserve > window:
            logger.warning(
                "A step's %d tool results, cut to their shares, still leave a %d-token "
                "request over a %d-token window with %d kept for the reply",
                len(tail),
                fitted,
                window,
                reserve,
            )
            return STEP_OVER_WINDOW_MESSAGE
        return None

    def _withhold_refused_step(self, live: _LiveSession) -> int:
        """Take a refused step out of the conversation, and tell the next turn (#1509).

        A refused step's results were never sent, and its `ASSISTANT` message was the
        model's reply, never sent back as input; so removing both rewrites nothing a
        model saw, as the dangling-step drop in `execute_turn` does not (#1423). Kept,
        the step stayed in every later request: turn-start compaction only shortens it,
        and past about a hundred parallel results on an 8K window the next plain turn
        went out over the window. Its calls ran, and may have changed something, so they
        are stated to the next turn the way a rolled-back attempt's are (#1495).

        Returns the number of calls withheld. They are the turn's last calls, and the
        turn leaves them out of `last_turn_tool_calls`, so a rollback of the same turn
        does not state them twice. That is done by position, not by call id: ids are not
        unique across steps or attempts (Ollama and Gemini number them `call_0`, `call_1`
        per response, and an id-less server gives `""`), so matching on them would drop
        other calls that ran.
        """
        history = live.messages
        start = unseen_step_start(history)
        if start is None:
            return 0
        withheld = history[start].tool_calls
        live.undone_tool_calls.extend(withheld)
        live.undone_tool_calls_shown = False
        del history[start:]
        live.updated_at = _now_iso()
        logger.warning(
            "Withheld a refused step (%d tool calls) from session %s of agent %s",
            len(withheld),
            self._context.session_id,
            self.agent_id,
        )
        return len(withheld)

    async def _execute_tools(
        self,
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest],
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord]]:
        """Execute a sequence of tool calls concurrently with P3 sandbox containment and path safety."""
        if not tool_calls or self._tools is None:
            return [], []

        workspace_root = self._resolve_workspace_root()
        isolation = self._resolve_tool_isolation()

        tool_ctx = ToolContext(
            agent_id=self.agent_id,
            session_id=self._context.session_id,
            trace_id=self._context.trace_id,
            workspace_root=workspace_root,
            read_roots=self._config.read_roots,
            isolation=isolation,
            turn_index=self._turn_counter,
            agent_delegate=self,
        )

        if len(tool_calls) == 1:
            msg, rec = await self._execute_single_tool(
                tool_calls[0], tool_ctx, stream_callback=stream_callback
            )
            return [msg], [rec]

        # Concurrently execute multiple tool calls (Issue #185)
        results = await asyncio.gather(
            *(
                self._execute_single_tool(tc, tool_ctx, stream_callback=stream_callback)
                for tc in tool_calls
            )
        )
        tool_messages = [r[0] for r in results]
        tool_executions = [r[1] for r in results]
        return tool_messages, tool_executions

    def available_tools(self) -> list[ToolProtocol]:
        """The tools this agent actually has: what a turn offers the model, before scoping.

        `config.allowed_tools` is a list of permissions, not of tools. A name in it may have
        nothing registered behind it -- a memory tool on an agent built without a store, a
        file tool on a registry that has none -- and an empty list permits everything. This
        is the registry filtered by that list, less what this agent cannot run. A per-turn
        `tool_scoper` may narrow it further for one turn.
        """
        if self._tools is None:
            return []
        allowed = self._config.allowed_tools if self._config.allowed_tools else None
        held: list[ToolProtocol] = []
        for t in drop_shadowed_aliases(self._tools.list_tools(filter_names=allowed)):
            # Advertising an agent-bound tool this agent cannot resolve would offer a
            # capability whose every call answers "not found".
            if isinstance(t, AGENT_BOUND_TOOL_TYPES) and t.name not in self._agent_local_tools:
                continue
            # A tool the persona flags withhold is not offered. This is the hint; the
            # refusals in `_execute_single_tool` and `execute_tool_call` are the
            # enforcement (#1167).
            if self._capability_refusal(t) is not None:
                continue
            held.append(t)
        return held

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

    def _resolve_tool(self, name: str) -> ToolProtocol | None:
        """The instance this agent executes for `name`, agent-local binding first.

        A tool held in `_agent_local_tools` is bound to state only this agent may act on --
        its memory store, and its skill registry with the `_loaded_skills` set that records
        what it loaded. Resolving those from the registry instead would hand an agent
        whichever instance was registered first, which in the UI is another agent's.
        """
        local = self._agent_local_tools.get(name)
        if local is not None:
            return local
        candidate = self._tools.get(name) if self._tools is not None else None
        # An agent-bound *instance* this agent did not register is *another* agent's, and
        # the shared registry will happily hand it over. An agent composed without a
        # memory store then records into whichever store was composed first; an agent
        # composed without a skill registry loads from another agent's approval list.
        # "No such tool" is the honest answer; a working call into someone else's state is
        # not (P6). A tool that merely shares the name is not one of those instances and
        # resolves normally.
        if isinstance(candidate, AGENT_BOUND_TOOL_TYPES):
            return None
        return candidate

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
        if self._resolve_tool(name) is None:
            raise KeyError(f"Tool '{name}' is not registered on agent '{self.agent_id}'")
        allowed = self._config.allowed_tools
        if allowed and name not in allowed:
            raise PermissionError(
                f"Tool '{name}' is registered on agent '{self.agent_id}' but is not in its "
                f"allowed_tools {tuple(allowed)!r}"
            )
        refusal = self._capability_refusal(self._resolve_tool(name))
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
            "tool_scoper": self._tool_scoper,
            "plan_generator": self._plan_generator,
            "workspace": self._host.workspace,
            "sandbox": self._host.sandbox,
            "isolation_floor": self._host.isolation_floor,
            "available_isolation": self._host.available_isolation,
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
        # Enforcement is inherited -- the parent's hooks, its budget, its tool scoping and
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
            sub_agent._agent_local_tools[reader.name] = reader
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
