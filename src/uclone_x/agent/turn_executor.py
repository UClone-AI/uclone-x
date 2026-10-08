"""The turn loop: one reasoning turn, its model calls, its tool rounds and its window fit.

Moved out of `agent/base.py` unchanged (#1736, stage 3). `BaseAgent` stays the facade:
`execute_turn` is its public entry point and delegates here, and it keeps `_invoke_model`,
`_fit_step_to_window` and `_execute_tools` as delegators, because tests call them and patch
them on the instance. The loop calls those three back *through the agent*, so a patched
one is the one a turn runs.

The executor holds no state of its own. Everything it reads or writes on the agent -- the
connector, the config, the live session, the turn counter, the step count, the fields the
tool context carries -- goes through a `TurnScope` of callables evaluated on every access,
so an agent whose `_llm`, `_tools`, `_budget` or live session is swapped after construction
is read as it now is. The accessors carry the names the agent's attributes have, so the
moved code reads as it did on `BaseAgent`.

The module-level helpers are the pure half: how a failed turn is classified
(`_turn_failure`), what a streamed call had produced when it stopped (`StreamProgress`),
and the two reads of history a turn makes before it starts (`_repeats_unanswered_prompt`,
`_unanswered_tool_step`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from uclone_x.agent import nudges
from uclone_x.agent.grounding import describe, unsupported_specifics
from uclone_x.agent.hooks import (
    HookAction,
    HookContext,
    HookEvent,
    HookRunner,
)
from uclone_x.agent.image_set_planner import is_planned_set
from uclone_x.agent.k_act import (
    TEXT_CALL_UNREADABLE_MESSAGE,
    TextCallStream,
    TextToolCallUnreadableError,
    text_call_response,
    text_tools_block,
)
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentState,
    ProviderFailure,
    ToolExecutionRecord,
    TurnResult,
    TurnStopReason,
)
from uclone_x.agent.nudges import (
    GROUNDING_REQUIRED_NUDGE_PREFIX,
    GROUNDING_REQUIRED_NUDGE_SUFFIX,
    apply_artifact_sanitization,
    evaluate_artifact_nudge,
    extract_produced_artifact_paths,
    grounding_supports,
    is_evidence_nudge_declined,
)
from uclone_x.agent.planner import ExecutionIntent, PlanGenerator
from uclone_x.agent.prompt_assembler import (
    PromptAssembler,
    drop_once_drawn,
    present_sections,
    undone_attempt_section,
)
from uclone_x.agent.protocols import (
    TurnAidHookProtocol,
    TurnAidProtocol,
    TurnLifecycleHookProtocol,
)
from uclone_x.agent.reply_lines import is_korean, lines_to_add, reply_notes, with_lines
from uclone_x.agent.request_record import (
    RequestLayers,
    assemble_request_messages,
)
from uclone_x.agent.session import (
    CompactionResult,
    redact_message,
)
from uclone_x.agent.text_tool_calls import detect_text_emitted_tool_calls
from uclone_x.agent.tool_execution import tool_outcome_of
from uclone_x.agent.tool_invoker import BoundToolsSession, ToolInvoker
from uclone_x.core.context_state import (
    EPOCH_PERSONA_EDITED,
    ContextForm,
    render_conversation,
)
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.core.secrets import redact_credentials
from uclone_x.core.session_log import (
    LoggedMessage,
    SessionLogEntry,
    SessionLogKind,
    logged_text,
    result_handle_of,
)
from uclone_x.core.tool_results import (
    STEP_REFUSAL_TEXT,
    STEP_REPLY_RESERVE_TOKENS,
    TOOL_RESULT_READ_TOOL,
    ResultBodies,
    StepRefusalCode,
    canonical_tool_text,
    result_handle,
    step_result_caps,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventPriority,
    EventType,
)
from uclone_x.engine.protocols import (
    EventBusProtocol,
    PublisherHandleProtocol,
)
from uclone_x.errors import (
    BudgetExceededError,
    LLMConnectorNotConfiguredError,
    LLMModelNotConfiguredError,
    LLMStreamInterruptedError,
    LLMTimeoutError,
    ModelLacksToolSupportError,
    ProviderFailureError,
    ProviderFailureKind,
    TokenBudgetExhaustedError,
    UsageLimitReachedError,
)
from uclone_x.llm.compactor import (
    estimate_message_tokens,
    estimate_reply_tokens,
    estimate_request_tokens,
    estimate_text_tokens,
    unseen_step_start,
)
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
    aggregate_token_usages,
)
from uclone_x.llm.protocols import (
    ImageInputProbe,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
    request_for_model,
)
from uclone_x.llm.router import LLMTier, SemanticModelRouter
from uclone_x.sandbox.models import (
    IsolationPolicy,
)
from uclone_x.skills.models import SkillStatus
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.telemetry.models import SpanStatus
from uclone_x.telemetry.protocols import TracerProtocol
from uclone_x.telemetry.tracer import FAILOVER_EVENT_SPAN_NAME
from uclone_x.tools.base import tool_returns_images
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.outcome import (
    ToolOutcome,
)
from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = [
    "PERSONA_EDIT_NOT_APPLIED",
    "StreamProgress",
    "TurnExecutor",
    "TurnScope",
    "TurnSession",
    "named_model",
]

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


#: How far up a `__cause__` chain `_is_provider_timeout` will look. A bound rather than a
#: `while`, because an exception chain can be made cyclic and a turn's error handler is
#: the last place that should be able to hang.
_MAX_CAUSE_DEPTH = 10


def named_model(value: object) -> str | None:
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
class StreamProgress:
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


#: What a turn whose saved persona edit could not be applied says in `error` (#1904). Fixed,
#: so no exception text, class name or path from the cause reaches a head; the cause is
#: logged. Heads word it in the reader's language from `stop_reason` instead.
PERSONA_EDIT_NOT_APPLIED = (
    "The changes saved to this clone could not be applied, so it did not answer. "
    "It kept its previous definition in this conversation; save the changes again to apply them."
)


class _PersonaEditNotApplied(Exception):
    """A staged persona edit that raised while this turn applied it (#1904).

    Raised from the cause, and carrying only `PERSONA_EDIT_NOT_APPLIED`, so every channel a
    failed turn reports through -- `error`, the ON_ERROR hook, the failover span -- carries
    the plain sentence, and the cause stays on `__cause__` for the log.
    """

    def __init__(self) -> None:
        super().__init__(PERSONA_EDIT_NOT_APPLIED)


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
) -> tuple[TurnStopReason, str, ProviderFailure | None]:
    """The stop reason, `error` and `provider_failure` a turn that raised `exc` ends with,
    logged as it deserves.

    A function of its own because `execute_turn` is at the edge of what the type checker
    can analyse, and each branch added inline pushes it over.
    """
    if _is_provider_timeout(exc):
        stop_reason = "provider_timeout"
    lacking_tools = _model_lacking_tools(exc)
    if lacking_tools is not None:
        # A choice of model, not a fault in the code. With no handler set up, Python prints
        # WARNING and above -- traceback included -- on the user's terminal, beside the
        # plain sentence the CLI already shows.
        logger.info("Turn for agent %s refused: %s", agent_id, lacking_tools)
        return "model_without_tools", str(lacking_tools), None
    provider_failure = _in_cause_chain(exc, ProviderFailureError)
    if provider_failure is not None:
        # Not a fault in the code either, and the raw response was logged where it was
        # classified; a traceback here would put it on the user's terminal a second time.
        logger.info("Turn for agent %s failed at the provider: %s", agent_id, provider_failure)
        failure = ProviderFailure.of(provider_failure)
        return _PROVIDER_STOP_REASONS[failure.kind], failure.message, failure
    if isinstance(exc, TextToolCallUnreadableError):
        # The model's reply, not a fault in the code: the reply is in the log's
        # MODEL_RESPONSE event, and the turn ends with the plain sentence (#2188).
        logger.info("Turn for agent %s refused: a tool call it wrote could not be read", agent_id)
        return "tool_call_unreadable", TEXT_CALL_UNREADABLE_MESSAGE, None
    if isinstance(exc, _PersonaEditNotApplied):
        # The cause is ours, not the provider's, so it is logged in full -- and only there.
        logger.error(
            "A saved persona edit could not be applied for agent %s; it was dropped",
            agent_id,
            exc_info=exc.__cause__,
        )
        return "persona_edit_failed", PERSONA_EDIT_NOT_APPLIED, None
    no_model = _in_cause_chain(exc, LLMModelNotConfiguredError)
    if no_model is not None:
        # A setting left empty, refused before the network: the stop reason is the model
        # being unavailable, which every head already words as "pick a model".
        logger.info("Turn for agent %s refused: %s", agent_id, no_model)
        failure = ProviderFailure(
            kind=ProviderFailureKind.MODEL_UNAVAILABLE,
            message=str(no_model),
            retryable=False,
            provider=no_model.provider,
        )
        return "model_unavailable", failure.message, failure
    logger.exception("Error executing turn for agent %s", agent_id)
    return stop_reason, str(exc), None


#: The stop reason each provider failure ends a turn with -- spelled as the kind is, and
#: written out so the type checker holds each one to `TurnStopReason`.
_PROVIDER_STOP_REASONS: dict[ProviderFailureKind, TurnStopReason] = {
    ProviderFailureKind.MODEL_UNAVAILABLE: "model_unavailable",
    ProviderFailureKind.PROVIDER_AUTH: "provider_auth",
    ProviderFailureKind.PROVIDER_QUOTA: "provider_quota",
    ProviderFailureKind.PROVIDER_UNREACHABLE: "provider_unreachable",
    ProviderFailureKind.PROVIDER_OUTAGE: "provider_outage",
    ProviderFailureKind.PROVIDER_ERROR: "provider_error",
}


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
    another in a transcript that names them (`ChatMessage.name`). Only a sender's own
    unanswered prompt is a repeat of it: were a head to name its users, two of them sending
    the same words would otherwise collapse into one.
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


_StreamCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class TurnSession(BoundToolsSession, Protocol):
    """The part of a live session a turn reads and writes, besides what the tools layer does."""

    updated_at: str
    last_turn_tool_calls: list[ToolCallRequest]
    undone_tool_calls: list[ToolCallRequest]
    undone_tool_calls_shown: bool
    recalled_memory: str | None
    results: dict[str, str]

    @property
    def messages(self) -> Sequence[ChatMessage]:
        """The session's history, each form rendered from its kept result (#1848)."""
        ...

    def append(self, *messages: ChatMessage) -> None:
        """Add `messages` to the history, each logged as it enters (#1443, #1848)."""
        ...

    def replace(self, index: int, message: ChatMessage, *, cause: str) -> None:
        """Rewrite the history message at `index`, declaring `cause` if it was shown."""
        ...

    def truncate(self, length: int, *, cause: str) -> None:
        """Cut the history to `length` messages, declaring `cause` if any was shown."""
        ...

    def declare_new_epoch(self, cause: str) -> None:
        """Say the next request may not extend what the last one showed (§5.8, Rule 1)."""
        ...

    def shown_in_epoch(self, index: int) -> bool:
        """Whether the history message at `index` is an entry the current epoch shows."""
        ...

    def log_entry(self, rendered: LoggedMessage) -> SessionLogEntry:
        """Log something a request sent that is not a history message (#1849)."""
        ...


class _PrepareTurnLayers(Protocol):
    def __call__(self, extra_sections: Sequence[str] = ...) -> RequestLayers: ...


class _PrepareTurnMessages(Protocol):
    def __call__(self, extra_sections: Sequence[str] = ...) -> list[ChatMessage]: ...


class _ImageSetSection(Protocol):
    def __call__(
        self,
        message: str,
        tool_defs: Sequence[ToolDefinition],
        *,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = ...,
    ) -> Awaitable[str | None]: ...


class _AutoCompact(Protocol):
    def __call__(
        self,
        tools: Sequence[ToolDefinition] = ...,
        extra_sections: Sequence[str] = ...,
        *,
        reason: str = ...,
    ) -> Awaitable[CompactionResult | None]: ...


class _IngestToolMessage(Protocol):
    def __call__(
        self, msg: ChatMessage, *, readable: bool, cap_bytes: int = ...
    ) -> ChatMessage: ...


class _ExecuteSingleTool(Protocol):
    def __call__(
        self,
        tc: ToolCallRequest,
        tool_ctx: ToolContext,
        *,
        stream_callback: _StreamCallback | None = ...,
        advertised: frozenset[str] | None = ...,
    ) -> Awaitable[tuple[ChatMessage, ToolExecutionRecord]]: ...


class _InvokeModel(Protocol):
    def __call__(
        self,
        llm: LLMProviderProtocol,
        req: LLMRequest,
        *,
        stream_callback: _StreamCallback | None = ...,
        progress: StreamProgress | None = ...,
    ) -> Awaitable[ModelResponse]: ...


class _FitStepToWindow(Protocol):
    def __call__(
        self,
        step_results: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        extra_sections: Sequence[str],
        *,
        readable: bool,
    ) -> StepRefusalCode | None: ...


class _ExecuteTools(Protocol):
    def __call__(
        self,
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest],
        *,
        stream_callback: _StreamCallback | None = ...,
        advertised: frozenset[str] | None = ...,
    ) -> Awaitable[tuple[list[ChatMessage], list[ToolExecutionRecord]]]: ...


@dataclass(frozen=True, slots=True)
class TurnScope:
    """What a `TurnExecutor` reads and writes on the agent it serves, each read when needed.

    Callables rather than values, because a turn is where the agent's state moves: the
    connector and the registry can be replaced after construction, a session switch
    changes the live history, and the turn itself advances the counter and the step
    count. The agent's own methods are reached through getters that return the agent's
    *current* bound method, so one a test or a subclass replaces on the instance is the
    one the turn calls.
    """

    #: The agent's id.
    agent_id: Callable[[], str]
    #: The injected LLM connector, or `None` when none is wired.
    llm: Callable[[], LLMProviderProtocol | None]
    #: The agent's `AgentConfig`.
    config: Callable[[], AgentConfig]
    #: The agent's `AgentContext`: session, trace and current state.
    context: Callable[[], AgentContext]
    #: The agent's tracer.
    tracer: Callable[[], TracerProtocol]
    #: The token budget manager, or `None`.
    budget: Callable[[], TokenBudgetManagerProtocol | None]
    #: The agent's tool registry, or `None` for an agent with no tools.
    tool_registry: Callable[[], ToolRegistryProtocol | None]
    #: The skill registry, or `None`.
    skills: Callable[[], SkillRegistryProtocol | None]
    #: The names of the skills this agent has loaded.
    loaded_skills: Callable[[], set[str]]
    #: The agent's lifecycle state.
    state: Callable[[], AgentState]
    #: The lock that serializes the agent's turns.
    turn_lock: Callable[[], asyncio.Lock]
    #: The agent's hook runner.
    hook_runner: Callable[[], HookRunner]
    #: The agent's tool invoker.
    tool_invoker: Callable[[], ToolInvoker]
    #: The agent's prompt assembler.
    prompt_assembler: Callable[[], PromptAssembler]
    #: The semantic model router, or `None`.
    semantic_router: Callable[[], SemanticModelRouter | None]
    #: The plan generator, or `None`.
    plan_generator: Callable[[], PlanGenerator | None]
    #: The agent's bus publisher handle, or `None` without a bus.
    publisher: Callable[[], PublisherHandleProtocol | None]
    #: The agent's event bus, or `None`.
    bus: Callable[[], EventBusProtocol | None]
    #: The active session's history, derived from its log: read-only (#1848). A turn
    #: writes it through the session's door (`TurnSession.append` and its siblings).
    history: Callable[[], Sequence[ChatMessage]]
    #: The active session's working copy.
    active_session: Callable[[], TurnSession]
    #: The agent's queue of durable events awaiting the store.
    pending_durable_events: Callable[[], list[dict[str, Any]]]
    #: The persona's display name, or `None`.
    persona_name: Callable[[], str | None]
    #: The persona's text, or `None`.
    persona: Callable[[], str | None]
    #: The agent's workspace root, or `None`.
    workspace_root: Callable[[], Path | None]
    #: What a tool context carries as `agent_delegate`: the agent itself.
    agent_delegate: Callable[[], Any]
    #: The turn counter, and its setter.
    turn_counter: Callable[[], int]
    set_turn_counter: Callable[[int], None]
    #: The steps the running turn has taken, and its setter.
    run_steps: Callable[[], int]
    set_run_steps: Callable[[int], None]
    #: The running turn's room, caller turn and story, and their setters.
    turn_room_id: Callable[[], str | None]
    set_turn_room_id: Callable[[str | None], None]
    turn_caller_turn_id: Callable[[], str | None]
    set_turn_caller_turn_id: Callable[[str | None], None]
    turn_story_id: Callable[[], str | None]
    set_turn_story_id: Callable[[str | None], None]
    #: Binds the running turn's workspace (`None`: the agent's own).
    set_turn_workspace_root: Callable[[Path | None], None]
    #: The agent's methods the turn calls back, as getters of the current bound method.
    transition_to: Callable[[], Callable[[AgentState], None]]
    live_session: Callable[[], Callable[[str], TurnSession]]
    result_bodies: Callable[[], Callable[[str], ResultBodies]]
    active_skill_dirs: Callable[[], Callable[[], tuple[Path, ...]]]
    context_window: Callable[[], Callable[[], int | None]]
    reply_reserve: Callable[[], Callable[[], int]]
    resolve_workspace_root: Callable[[], Callable[[], Path | None]]
    resolve_tool_isolation: Callable[[], Callable[[], IsolationPolicy]]
    prepare_turn_layers: Callable[[], _PrepareTurnLayers]
    prepare_turn_messages: Callable[[], _PrepareTurnMessages]
    nudged_retry: Callable[
        [], Callable[[LLMRequest, str, Sequence[str]], tuple[LLMRequest, RequestLayers]]
    ]
    image_set_section: Callable[[], _ImageSetSection]
    case_skill_section: Callable[[], _ImageSetSection]
    auto_compact_if_needed: Callable[[], _AutoCompact]
    ingest_tool_message: Callable[[], _IngestToolMessage]
    execute_single_tool: Callable[[], _ExecuteSingleTool]
    after_tool_step: Callable[[], Callable[[Sequence[ToolExecutionRecord], ToolContext], None]]
    #: The agent's lifecycle hooks; those that are `TurnAidHookProtocol` add turn aids.
    lifecycle_hooks: Callable[[], Sequence[TurnLifecycleHookProtocol]]
    invoke_model: Callable[[], _InvokeModel]
    fit_step_to_window: Callable[[], _FitStepToWindow]
    execute_tools: Callable[[], _ExecuteTools]
    take_staged_persona: Callable[[], Callable[[], Callable[[], None] | None]]


class TurnExecutor:
    """Runs one agent's turns: the model calls, the tool rounds and the window fit."""

    def __init__(self, scope: TurnScope) -> None:
        self._scope = scope
        #: The running turn's person's names, for its tool calls (#1857).
        self._turn_person_names: tuple[str, ...] = ()
        #: The running turn's aids, worked out once at its start (#1808).
        self._turn_aids: tuple[TurnAidProtocol, ...] = ()
        #: The last tool step's result handles, by call id, for its `TOOL_RESULT` events.
        self._step_result_handles: dict[str, str] = {}

    # -- the agent's state, read through the scope ---------------------------------

    @property
    def agent_id(self) -> str:
        return self._scope.agent_id()

    @property
    def _llm(self) -> LLMProviderProtocol | None:
        return self._scope.llm()

    @property
    def _config(self) -> AgentConfig:
        return self._scope.config()

    @property
    def _context(self) -> AgentContext:
        return self._scope.context()

    @property
    def _tracer(self) -> TracerProtocol:
        return self._scope.tracer()

    @property
    def _budget(self) -> TokenBudgetManagerProtocol | None:
        return self._scope.budget()

    @property
    def _tools(self) -> ToolRegistryProtocol | None:
        return self._scope.tool_registry()

    @property
    def _skills(self) -> SkillRegistryProtocol | None:
        return self._scope.skills()

    @property
    def _loaded_skills(self) -> set[str]:
        return self._scope.loaded_skills()

    @property
    def _state(self) -> AgentState:
        return self._scope.state()

    @property
    def _turn_lock(self) -> asyncio.Lock:
        return self._scope.turn_lock()

    @property
    def _hook_runner(self) -> HookRunner:
        return self._scope.hook_runner()

    @property
    def _tool_invoker(self) -> ToolInvoker:
        return self._scope.tool_invoker()

    @property
    def _prompt_assembler(self) -> PromptAssembler:
        return self._scope.prompt_assembler()

    @property
    def _semantic_router(self) -> SemanticModelRouter | None:
        return self._scope.semantic_router()

    @property
    def _plan_generator(self) -> PlanGenerator | None:
        return self._scope.plan_generator()

    @property
    def _publisher(self) -> PublisherHandleProtocol | None:
        return self._scope.publisher()

    @property
    def _bus(self) -> EventBusProtocol | None:
        return self._scope.bus()

    @property
    def _history(self) -> Sequence[ChatMessage]:
        return self._scope.history()

    @property
    def _active_session(self) -> TurnSession:
        return self._scope.active_session()

    @property
    def _pending_durable_events(self) -> list[dict[str, Any]]:
        return self._scope.pending_durable_events()

    @property
    def persona_name(self) -> str | None:
        return self._scope.persona_name()

    @property
    def persona(self) -> str | None:
        return self._scope.persona()

    @property
    def workspace_root(self) -> Path | None:
        return self._scope.workspace_root()

    @property
    def _agent_delegate(self) -> Any:
        return self._scope.agent_delegate()

    @property
    def _turn_counter(self) -> int:
        return self._scope.turn_counter()

    @_turn_counter.setter
    def _turn_counter(self, value: int) -> None:
        self._scope.set_turn_counter(value)

    @property
    def _run_steps(self) -> int:
        return self._scope.run_steps()

    @_run_steps.setter
    def _run_steps(self, value: int) -> None:
        self._scope.set_run_steps(value)

    @property
    def _turn_room_id(self) -> str | None:
        return self._scope.turn_room_id()

    @_turn_room_id.setter
    def _turn_room_id(self, value: str | None) -> None:
        self._scope.set_turn_room_id(value)

    @property
    def _turn_caller_turn_id(self) -> str | None:
        return self._scope.turn_caller_turn_id()

    @_turn_caller_turn_id.setter
    def _turn_caller_turn_id(self, value: str | None) -> None:
        self._scope.set_turn_caller_turn_id(value)

    @property
    def _turn_story_id(self) -> str | None:
        return self._scope.turn_story_id()

    @_turn_story_id.setter
    def _turn_story_id(self, value: str | None) -> None:
        self._scope.set_turn_story_id(value)

    # -- the agent's methods, called back through it -------------------------------

    @property
    def transition_to(self) -> Callable[[AgentState], None]:
        return self._scope.transition_to()

    @property
    def _live_session(self) -> Callable[[str], TurnSession]:
        return self._scope.live_session()

    @property
    def _result_bodies(self) -> Callable[[str], ResultBodies]:
        return self._scope.result_bodies()

    @property
    def active_skill_dirs(self) -> Callable[[], tuple[Path, ...]]:
        return self._scope.active_skill_dirs()

    @property
    def _context_window(self) -> Callable[[], int | None]:
        return self._scope.context_window()

    @property
    def _reply_reserve(self) -> Callable[[], int]:
        return self._scope.reply_reserve()

    @property
    def _resolve_workspace_root(self) -> Callable[[], Path | None]:
        return self._scope.resolve_workspace_root()

    @property
    def _resolve_tool_isolation(self) -> Callable[[], IsolationPolicy]:
        return self._scope.resolve_tool_isolation()

    @property
    def _prepare_turn_layers(self) -> _PrepareTurnLayers:
        return self._scope.prepare_turn_layers()

    @property
    def _prepare_turn_messages(self) -> _PrepareTurnMessages:
        return self._scope.prepare_turn_messages()

    @property
    def _nudged_retry(
        self,
    ) -> Callable[[LLMRequest, str, Sequence[str]], tuple[LLMRequest, RequestLayers]]:
        return self._scope.nudged_retry()

    @property
    def _image_set_section(self) -> _ImageSetSection:
        return self._scope.image_set_section()

    @property
    def _case_skill_section(self) -> _ImageSetSection:
        return self._scope.case_skill_section()

    async def _turn_start_sections(
        self,
        turn_live: TurnSession,
        message: str,
        tool_defs: Sequence[ToolDefinition],
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> tuple[list[str], str | None]:
        """The turn's extra tail sections, and the image-set plan among them, if any.

        Also recalls the memory section for `message` onto `turn_live`. Kept out of the
        turn loop, whose body is at the type checker's complexity limit.
        """
        # What an undone attempt called, stated once per retry and kept for every
        # step of it (#1495). Cleared when the turn that showed it was not rolled
        # back: that turn is now in history and speaks for itself.
        if turn_live.undone_tool_calls_shown:
            turn_live.undone_tool_calls.clear()
            turn_live.undone_tool_calls_shown = False
        undone_section = undone_attempt_section(turn_live.undone_tool_calls)
        turn_live.undone_tool_calls_shown = bool(undone_section)
        self._announce_bound_on_message(turn_live, tool_defs)
        image_set_section, case_section = await self._drawing_sections(message, tool_defs, emit)
        # Recall, once per turn and before compaction counts the request: the clone's
        # facts ranked against this message (clone-knowledge-graph §3.5). Held on the
        # session so every step of the turn sends the same tail.
        turn_live.recalled_memory = await self._prompt_assembler.recall_memory(message)
        self._log_recalled_memory(turn_live)
        self._turn_aids = self._work_out_turn_aids(message)
        sections = list(present_sections(undone_section, image_set_section, case_section))
        sections.extend(aid.section for aid in self._turn_aids)
        return sections, image_set_section

    def _announce_bound_on_message(
        self, turn_live: TurnSession, tool_defs: Sequence[ToolDefinition]
    ) -> None:
        """``k_act`` (#2188): describe the tools this message bound on the message itself.

        Appended to the user message the turn just added, which no request has shown yet,
        so the rewrite declares no epoch: the next request only extends the last one. The
        description stays in history, so it is sent once and repeated as the same bytes.
        Nothing happens under another module, or when nothing new was bound.
        """
        block = text_tools_block(
            self._tool_invoker.text_tools_to_announce(tool_defs, turn_live.messages)
        )
        history = turn_live.messages
        if block is None or not history or history[-1].role is not MessageRole.USER:
            return
        last = len(history) - 1
        prompt = history[last]
        announced = prompt.model_copy(update={"content": f"{prompt.content}\n\n{block}"})
        turn_live.replace(last, announced, cause="text_tools_bound")

    def _announce_bound_on_call(
        self,
        tool_messages: list[ChatMessage],
        step_executions: Sequence[ToolExecutionRecord],
        shown_tool_names: set[str],
    ) -> None:
        """``k_act`` (#2188): describe a tool bound on the call that named it, on its result.

        A catalog tool the step ran without having been offered (R3, bind on call) is
        described once, appended to that call's result, before the result enters history.
        """
        live = self._live_session(self._context.session_id)
        bound_now = [
            d
            for d in self._tool_invoker.session_tools_layer(live)
            if d.name not in shown_tool_names
        ]
        fresh = self._tool_invoker.text_tools_to_announce(bound_now, live.messages)
        for index, (message, record) in enumerate(
            zip(tool_messages, step_executions, strict=False)
        ):
            tools = [d for d in fresh if d.name == record.tool_name]
            block = text_tools_block(tools)
            if block is not None and message.content is not None:
                tool_messages[index] = message.model_copy(
                    update={"content": f"{message.content}\n\n{block}"}
                )
                fresh = [d for d in fresh if d.name != record.tool_name]

    def _work_out_turn_aids(self, message: str) -> tuple[TurnAidProtocol, ...]:
        """The aids the host's `TurnAidHookProtocol` hooks give this turn (#1808).

        Worked out once, before the turn's first request, and kept for every step: the
        sections go in the turn context, which does not change between steps. A hook that
        raises is logged and gives nothing; the turn runs without it.
        """
        aids: list[TurnAidProtocol] = []
        tool_names = frozenset(t.name for t in self._tool_invoker.available_tools())
        story_id, room_id = self._turn_story_id, self._turn_room_id
        for hook in self._scope.lifecycle_hooks():
            if not isinstance(hook, TurnAidHookProtocol):
                continue
            try:
                aid = hook.turn_aid(
                    message=message,
                    story_id=story_id,
                    room_id=room_id,
                    workspace_root=self._resolve_workspace_root(),
                    tool_names=tool_names,
                )
            except Exception:
                logger.warning(
                    "Agent %s: a turn aid failed; the turn runs without it",
                    self.agent_id,
                    exc_info=True,
                )
                continue
            if aid is not None:
                aids.append(aid)
        return tuple(aids)

    def _late_nudge(
        self,
        content: str,
        called_tools: bool,
        nudged: bool,
        asked: Collection[str],
        step: int,
        durable_events: list[dict[str, Any]],
        tool_executions: Sequence[ToolExecutionRecord] = (),
        user_message: str | None = None,
        advertised_tool_names: Sequence[str] = (),
    ) -> str | None:
        """The last nudge a step's answer can get: an empty reply's, or a missing image's.

        Once per turn between them (`nudged` is the turn's latch). A step with no content
        and no tool call is asked again for a reply (#1808): qwen3:8b ended 3 of 8 Writer
        turns that way. Only while no nudge of any kind has been sent this turn (`asked`):
        an empty answer to the evidence or grounding nudge already keeps the first answer. A step cut off by the output limit never gets here (it raises
        `TokenBudgetExhaustedError`), so an empty reply here had room left to write, and a
        second one stops the turn as it did before. Otherwise an answer that links an image
        no tool made is asked to make it (`evaluate_artifact_nudge`). Records the nudge's
        durable event; returns its text, or `None` when there is none to give.
        """
        if not asked and not content and not called_tools:
            durable_events.append(
                {"type": "EMPTY_REPLY_NUDGE", "step": step, "turn_index": self._turn_counter}
            )
            return nudges.compose_empty_reply_nudge(advertised_tool_names)
        produced_paths = extract_produced_artifact_paths(tool_executions)
        art_nudge_info = evaluate_artifact_nudge(
            content,
            self._tools,
            self.workspace_root,
            nudged,
            produced_paths=produced_paths,
            user_message=user_message,
        )
        if art_nudge_info is None:
            return None
        missing_path, artifact_nudge = art_nudge_info
        durable_events.append(
            {
                "type": "ARTIFACT_NUDGE",
                "step": step,
                "turn_index": self._turn_counter,
                "missing_artifact": missing_path,
            }
        )
        return artifact_nudge

    def _sanitize_turn_artifacts(
        self,
        resp_content: str,
        assistant_msg_idx: int | None,
    ) -> str:
        """Sanitize missing image links after the turn completes."""
        # ``k_act`` (#2188): an answer that ends in a cut-off opening tag loses it.
        shown = self._tool_invoker.visible_reply(resp_content)
        sanitized = apply_artifact_sanitization(shown, self.workspace_root)
        if (
            sanitized != resp_content
            and assistant_msg_idx is not None
            and assistant_msg_idx < len(self._history)
        ):
            self._rewrite_answer(assistant_msg_idx, sanitized, "artifact_sanitized")
        return sanitized

    async def _add_reply_lines(
        self,
        content: str,
        assistant_msg_idx: int | None,
        message: str,
        records: Sequence[ToolExecutionRecord],
        durable_events: list[dict[str, Any]],
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> str:
        """`content` with the lines code adds to a finished turn's reply (#1808).

        Each tool's reply note and each turn aid's line, in the language of `message`,
        once each and only when the reply does not already hold it (`agent/reply_lines.py`).
        The history's final answer is rewritten to match, or appended when the turn ended
        without one, and the added text is streamed as a token, so a streamed reply and the
        stored one agree.
        """
        korean = is_korean(message)
        wanted = [
            *reply_notes(records, korean=korean),
            *(line for aid in self._turn_aids for line in aid.reply_lines(records, korean=korean)),
        ]
        added = lines_to_add(content, wanted)
        if not added:
            return content
        full = with_lines(content, added)
        history = self._history
        if content and assistant_msg_idx is not None and assistant_msg_idx < len(history):
            self._rewrite_answer(assistant_msg_idx, full, "reply_lines")
        else:
            self._active_session.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))
        durable_events.append({"type": "REPLY_LINES_ADDED", "lines": added})
        await emit("token", {"content": full[len(content) :]})
        return full

    def _log_recalled_memory(self, turn_live: TurnSession) -> None:
        """Log the memory section this turn recalled as a `memory` entry (#1849).

        The section is sent in every request of the turn, in the `[Turn Context]` tail,
        and is not a history message, so nothing else logs it. The entry's body is the
        section itself and its handle names that body (#1848), so `tool_result_read`
        reads it back after the turn that sent it, from the session's own store. What the
        requests send is unchanged: the section is not rewritten and no handle is added
        to it.
        """
        section = turn_live.recalled_memory
        if not section:
            return
        handle = result_handle(redact_credentials(section))
        turn_live.log_entry(logged_text(SessionLogKind.MEMORY, section, blob=handle))

    async def _drawing_sections(
        self,
        message: str,
        tool_defs: Sequence[ToolDefinition],
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> tuple[str | None, str | None]:
        """The turn-start sections that steer drawing: an image-set plan, else case skills.

        Both go at the tail, after the user's message, so the prefix stays cacheable.
        The plan is kept until `generate_image` has run: a first step that reads a
        character sheet must not cost it. The case skills stay for every step of the
        turn. A planned set carries its own prompts, so it is not routed as well; a failed
        plan's fallback note carries none, so its turn is routed as one with no set would be.
        Kept out of the turn loop, whose body is at the type checker's complexity limit.
        """
        image_set_section = await self._image_set_section(message, tool_defs, emit=emit)
        if image_set_section is not None and is_planned_set(image_set_section):
            return image_set_section, None
        return image_set_section, await self._case_skill_section(message, tool_defs)

    @property
    def _auto_compact_if_needed(self) -> _AutoCompact:
        return self._scope.auto_compact_if_needed()

    def _take_staged_persona(self) -> None:
        """Apply a saved persona edit at this turn's start, and open an epoch for it.

        Runs under the turn lock, so an edit saved while a turn is in flight waits for
        the next one (#1899). The declared cause opens a new epoch at this turn's first
        request even though the history only grew: the identity layer changed here, and
        the session log records the boundary (`EPOCH_PERSONA_EDITED`).

        The edit applies whole or not at all (#1904). Taking it works out the new prompt
        and tool scope without applying either; the epoch is declared next; and only then
        does the returned step assign. A raise at any point before that leaves the seat on
        its old definition with no epoch opened, never the new prompt under the old tools.

        A raise is re-raised as `_PersonaEditNotApplied`, so the failed turn is named
        `persona_edit_failed` and tells the person who saved the edit in plain words, never
        in the cause's text (#1904).
        """
        try:
            apply_edit = self._scope.take_staged_persona()()
            if apply_edit is not None:
                self._active_session.declare_new_epoch(EPOCH_PERSONA_EDITED)
                apply_edit()
        except Exception as exc:
            raise _PersonaEditNotApplied from exc

    @property
    def _ingest_tool_message(self) -> _IngestToolMessage:
        return self._scope.ingest_tool_message()

    @property
    def _execute_single_tool(self) -> _ExecuteSingleTool:
        return self._scope.execute_single_tool()

    @property
    def _after_tool_step(
        self,
    ) -> Callable[[Sequence[ToolExecutionRecord], ToolContext], None]:
        return self._scope.after_tool_step()

    @property
    def _invoke_model(self) -> _InvokeModel:
        return self._scope.invoke_model()

    @property
    def _fit_step_to_window(self) -> _FitStepToWindow:
        return self._scope.fit_step_to_window()

    @property
    def _execute_tools(self) -> _ExecuteTools:
        return self._scope.execute_tools()

    def _append_step_results(
        self, step_results: Sequence[ChatMessage], reader_offered: bool
    ) -> None:
        """Append a step's redacted results to the history, each held to the cap (#1422).

        Notes the handle each result is read by, by call id: the log's `TOOL_RESULT`
        event names the result by it and holds no second copy of the text (#2013).
        """
        ingested = [self._ingest_tool_message(m, readable=reader_offered) for m in step_results]
        self._active_session.append(*ingested)
        self._step_result_handles = {
            m.tool_call_id: result_handle_of(m) for m in ingested if m.tool_call_id
        }

    def _result_handle_of(self, record: ToolExecutionRecord) -> str | None:
        """The handle the step's result for `record` is read by, or `None` (#2013)."""
        if record.tool_call_id is None:
            return None
        return self._step_result_handles.get(record.tool_call_id)

    # -- the turn ------------------------------------------------------------------

    async def invoke_model(
        self,
        llm: LLMProviderProtocol,
        req: LLMRequest,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        progress: StreamProgress | None = None,
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

        # Every step's request passes here -- a 1:1 turn, a room seat's, a retry's, and the
        # history a compaction left -- so a model that cannot see is never sent the images
        # an earlier model was shown (#2123). The stored history keeps them.
        req = await request_for_model(llm, req)

        if stream_callback is not None and hasattr(llm, "stream"):
            # What was asked for: the request's model, else the one the connector says it
            # sends a request naming none to. `None` when neither is knowable -- never the
            # literal "default", which is a placeholder no provider serves and which the
            # room then showed as `ollama:default` while `OLLAMA_MODEL` answered (#1447).
            requested_model = (
                named_model(req.model)
                or named_model(getattr(llm, "_default_model", None))
                or named_model(getattr(llm, "model", None))
            )
            # What served it: whatever the stream itself reports, like `generate` reads the
            # response body. The last chunk that names one wins.
            served_model: str | None = None
            if progress is None:
                progress = StreamProgress()
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
            except UsageLimitReachedError:
                # The usage gate refused before the first chunk: nothing was sent, so nothing
                # is booked, and the turn stops as `usage_limit`, not as an interrupted
                # stream (the token-gateway design §4.5.4).
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
        progress = StreamProgress()
        # ``k_act`` (#2188): the reply's calls are its text, which a person never watches.
        text_calls = self._tool_invoker.tools_module == "k_act"
        if text_calls and stream_callback is not None:
            stream_callback = TextCallStream(stream_callback)
        try:
            resp = await self._invoke_model(
                llm, req, stream_callback=stream_callback, progress=progress
            )
            if isinstance(stream_callback, TextCallStream):
                await stream_callback.flush()
        except (Exception, asyncio.CancelledError) as invoke_exc:
            ended_at = _now_iso()
            partial_content = getattr(invoke_exc, "partial_content", None)
            if not isinstance(partial_content, str):
                partial_content = progress.content if progress.content_chunks else None
            err_model = named_model(getattr(invoke_exc, "model", None)) or named_model(req.model)
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
        served_name = named_model(getattr(resp, "model_name", None))
        model_name = (served_name if served_name != "unknown" else None) or named_model(req.model)
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
        if text_calls:
            # Recorded above as the model wrote it; read here. A reply whose call cannot be
            # read raises the plain refusal, and nothing of it enters history (#2188).
            resp = text_call_response(resp)
        return resp

    async def execute_turn(
        self,
        input_data: str | AgentEvent,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        caller_turn_id: str | None = None,
        room_id: str | None = None,
        story_id: str | None = None,
        person_names: tuple[str, ...] = (),
        workspace_root: Path | None = None,
    ) -> TurnResult:
        """Run one reasoning turn; see `BaseAgent.execute_turn`, which delegates here."""
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
            self._turn_room_id = room_id
            self._turn_caller_turn_id = caller_turn_id
            self._turn_story_id = story_id
            # Set under the turn lock, so a clone seated in two rooms never reads the
            # other room's person (#1857).
            self._turn_person_names = person_names
            # Under the lock too: a turn queued behind this one cannot move its sandbox.
            self._scope.set_turn_workspace_root(workspace_root)
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
                # Inside the `try` (#1904): an edit that fails to apply is a failed turn
                # result, like any other failure at the turn's start, not an exception out
                # of `execute_turn`. It runs before the first request is built.
                self._take_staged_persona()
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
                    self._active_session.append(user_prompt)
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
                turn_live = self._live_session(turn_session_id)
                # The tools layer, settled once per user message (design §5.1): host
                # binding may append to it here. After that only a `search_tools` call
                # that finds a new tool changes it, by appending, from the next step on
                # (`ToolInvoker.tools_after_step`).
                tool_defs: list[ToolDefinition] = []
                if self._tools is not None:
                    tool_defs = await self._tool_invoker.tools_for_turn(content_input, turn_live)

                turn_extra_sections, image_set_section = await self._turn_start_sections(
                    turn_live, content_input, tool_defs, _emit_stream
                )

                # Compact before dispatch (Requirement 3). Placed after the user message is
                # appended, so the message that may itself tip the context over the
                # threshold is counted, and after the tool list is settled, so the count
                # is of the request this step sends -- system sections, turn context and
                # tool schemas included (#1422) -- and before `_prepare_turn_layers`,
                # so the request is built from the compacted sequence.
                compaction = await self._auto_compact_if_needed(tool_defs, turn_extra_sections)
                if compaction is not None and self._tools is not None:
                    # The compacted request starts a new prefix anyway, so bind this
                    # message again from the base set rather than carry the old set.
                    tool_defs = await self._tool_invoker.tools_for_turn(content_input, turn_live)

                req_layers = self._prepare_turn_layers(extra_sections=turn_extra_sections)
                turn_messages = assemble_request_messages(req_layers)
                req = LLMRequest(
                    model=self._config.llm_config.model_name or None,
                    messages=tuple(turn_messages),
                    tools=self._tool_invoker.declared_tools(tool_defs),
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
                tools_disallowed = False
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
                            content=self._tool_invoker.visible_reply(resp_content),
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
                            **self._prompt_assembler.request_context_fields(step, req, req_layers),
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
                            self._active_session.truncate(len(self._history) - 1, cause="retry")
                            self._active_session.declare_new_epoch("retry")
                        superseded = None
                        assistant_msg_idx = len(self._history)
                        # Logged as it enters, after the prompt: a retry, a declined nudge
                        # or a post-turn hook can take it out of the history again, not
                        # out of the log (#1443).
                        self._active_session.append(
                            ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=resp_content or None,
                                # This step's calls only. The message is the record of one
                                # invocation; the turn's accumulation belongs on TurnResult.
                                tool_calls=tool_calls,
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
                            tool_outcome_of(record) == ToolOutcome.PRODUCTIVE.value
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
                                grounding_supports(req.messages, tool_executions, injected_nudges),
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
                            injected_nudges.add(nudges.EVIDENCE_REQUIRED_NUDGE)
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
                                nudges.EVIDENCE_REQUIRED_NUDGE,
                                turn_extra_sections,
                            )
                            continue

                        if (
                            evidence_nudged
                            # Not merely `is not None`: an empty first answer never
                            # entered history, so there is nothing to fall back to, and
                            # "declining" would discard the retry's answer for "" (#1420).
                            and first_answer
                            and is_evidence_nudge_declined(resp_content, first_answer)
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
                            # The retry's answer was shown to no request of this turn,
                            # but the next turn's would have extended it (§5.8, Rule 1).
                            self._active_session.declare_new_epoch("retry")
                            if first_assistant_msg is not None:
                                if assistant_msg_idx is not None and assistant_msg_idx < len(
                                    self._history
                                ):
                                    self._active_session.replace(
                                        assistant_msg_idx, first_assistant_msg, cause="retry"
                                    )
                                else:
                                    assistant_msg_idx = len(self._history)
                                    self._active_session.append(first_assistant_msg)
                            elif assistant_msg_idx is not None and assistant_msg_idx < len(
                                self._history
                            ):
                                # The first answer was empty and so never entered history.
                                # The retry's answer is the step's last message: this step
                                # made no calls, or it would not be a final answer.
                                self._active_session.truncate(assistant_msg_idx, cause="retry")
                                assistant_msg_idx = None

                        # What the answer asserts that nothing this turn read contains.
                        # Computed on every turn, acted on only when the agent asked for
                        # it: the observation is what #700 compares against a declared
                        # horizon and what a fabrication signal can be built from now that
                        # traps have left the eval set (#733), and gating the field on the
                        # setting would withhold it from every agent that has not opted in.
                        unsupported = unsupported_specifics(
                            resp_content,
                            grounding_supports(req.messages, tool_executions, injected_nudges),
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

                        # Missing artifact image hallucination check (Tier 1 Nudge), or an
                        # empty reply asked again (#1808): one of the two, once per turn.
                        artifact_nudge = self._late_nudge(
                            resp_content,
                            bool(tool_calls),
                            artifact_nudged,
                            injected_nudges,
                            step,
                            durable_events,
                            tool_executions,
                            content_input,
                            advertised_tool_names=[d.name for d in tool_defs],
                        )
                        if artifact_nudge is not None:
                            artifact_nudged = True
                            injected_nudges.add(artifact_nudge)
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
                        elif evidence_nudged or artifact_nudged or tools_disallowed:
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
                    # Only what this request advertised runs (F12): a provider may pass a
                    # call to an undeclared name through, and the model was never shown it.
                    tool_messages, step_executions, guarded = await self._execute_step_tools(
                        tool_calls,
                        tool_executions,
                        req,
                        tool_defs,
                        stream_callback,
                        shown_tool_names,
                    )
                    if guarded:
                        tools_disallowed = True
                    # Redacted, then held to the result cap (#1422): an over-cap result is
                    # stored in full and the history keeps an excerpt naming it. Decided
                    # here, once; later steps render the same excerpt.
                    reader_offered = any(d.name == TOOL_RESULT_READ_TOOL for d in tool_defs)
                    step_results = [redact_message(m) for m in tool_messages]
                    drop_once_drawn(turn_extra_sections, image_set_section, step_executions)
                    self._append_step_results(step_results, reader_offered)
                    tool_executions.extend(step_executions)
                    tools_unreported = False
                    for tr in step_executions:
                        durable_events.append(
                            {
                                "type": "TOOL_RESULT",
                                "tool_call_id": tr.tool_call_id,
                                "name": tr.tool_name,
                                "result_handle": self._result_handle_of(tr),
                                # `status` alone cannot tell a search that answered from
                                # one that matched nothing -- both are "success" (#698).
                                # Without this the log shows a healthy call before a turn
                                # that gave up, and the giving up looks unmotivated.
                                "status": tr.status,
                                "outcome": tool_outcome_of(tr),
                                "at": _now_iso(),
                                "duration_ms": tr.duration_ms,
                            }
                        )
                    self.transition_to(AgentState.REASONING)
                    await _emit_stream(
                        "status",
                        {"status": "thinking", "detail": "Processing tool results..."},
                    )

                    # No compaction runs here (§5.8, owner ruling 2026-09-27). An epoch
                    # starts only at a turn boundary, so every request of this turn extends
                    # the one before it, and nothing the model was already shown is
                    # shrunk to make room for this step. This reverses #1422's pass
                    # between steps; a step that does not fit is refused below instead.
                    #
                    # The step's own results, together, must still fit the window (#1480).
                    # Each is under the result cap, but several can exceed what the
                    # history as this turn's requests sent it leaves; they are cut to
                    # excerpts of an equal share, or, when even that cannot fit, the step
                    # is refused and nothing over the window is sent.
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
                            error=STEP_REFUSAL_TEXT[step_refusal],
                            error_code=step_refusal,
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
                    # A `search_tools` hit is declared from the next request on (§5.1).
                    # One statement for both: execute_turn sits at pyright's
                    # flow-analysis limit, and one more statement here exceeds it.
                    req = self._tool_invoker.tools_after_step(
                        req, step_messages, turn_live, tool_defs, shown_tool_names, step_executions
                    )

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
                        "workspace_root": str(self.workspace_root) if self.workspace_root else None,
                        "user_message": content_input,
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
                            self._rewrite_answer(assistant_msg_idx, resp_content, "post_turn_hook")
                        elif resp_content:
                            assistant_msg_idx = len(self._history)
                            self._active_session.append(
                                ChatMessage(role=MessageRole.ASSISTANT, content=resp_content)
                            )

                resp_content = self._sanitize_turn_artifacts(resp_content, assistant_msg_idx)
                resp_content = await self._add_reply_lines(
                    resp_content,
                    assistant_msg_idx,
                    content_input,
                    tool_executions,
                    durable_events,
                    _emit_stream,
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
                if isinstance(exc, UsageLimitReachedError):
                    # The user's system-wide paid-model limit (`llm-token-gateway.md`
                    # §4.3), not this session's ceiling: a new conversation meets it too,
                    # and its message is written for the user, so heads show it as is.
                    stop_reason = "usage_limit"
                elif isinstance(exc, BudgetExceededError):
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
                stop_reason, failure, provider_failure = _turn_failure(
                    exc, stop_reason, self.agent_id
                )
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
                    provider_failure=provider_failure,
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
                    # Through the session's door: what leaves stays in the log, and a
                    # request that showed the step makes the next one open an epoch.
                    if kept_text:
                        dropped_count = len(turn_messages) - dangling_index - 1
                        turn_live.truncate(dangling_index + 1, cause="tool_step_dropped")
                        turn_live.replace(
                            dangling_index,
                            asked.model_copy(update={"tool_calls": ()}),
                            cause="tool_step_dropped",
                        )
                    else:
                        dropped_count = len(turn_messages) - dangling_index
                        turn_live.truncate(dangling_index, cause="tool_step_dropped")
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

    def _rewrite_answer(self, index: int, content: str, cause: str) -> None:
        """Rewrite the answer at `index` of the history to `content`, keeping its calls.

        Through the session's door (`replace`), which logs the new text and declares
        `cause` only when a request of this epoch showed the message: rewriting a message
        a request showed breaks Rule 1, so the next request opens an epoch naming `cause`.
        Rewriting the final answer no request has shown yet does not: the next request
        only appends it, and a cause declared for it would label the next epoch -- a
        compaction's -- with it (#1854).
        """
        self._active_session.replace(
            index,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content or None,
                tool_calls=self._history[index].tool_calls,
            ),
            cause=cause,
        )

    def fit_step_to_window(
        self,
        step_results: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
        extra_sections: Sequence[str],
        *,
        readable: bool,
    ) -> StepRefusalCode | None:
        """Cut the step that just ran to fit the window, or say why it cannot (#1480).

        `step_results` are the step's results before the cap, in the order appended; the
        history's trailing tool-call group holds them as ingested. The budget is the
        window less the room kept for the reply (`_reply_reserve`, #1509) and everything
        else the next request sends -- the system turn, the tool schemas, the turn
        context and the history as this turn's earlier requests sent it -- no pass between
        steps shrinks it (§5.8). When the step's results exceed it,
        `step_result_caps` shares it out and each over-share result is ingested again at
        its share: an excerpt whose full body stays readable with `tool_result_read`.
        Nothing a request has shown is rewritten: the model has not seen this step yet.

        `None` when the next request fits. Otherwise the code of the plain refusal the
        turn ends with (`STEP_REFUSAL_TEXT` holds each one's sentence, #1862):
        `step.no_room` when the request leaves no room for the step at
        all, so the conversation and not the tools is the cause (#1509);
        `step.no_room_setup` when the system turn and the tool schemas alone
        leave none, so no shortening would help (#1866), or
        `step.no_room_setup_reply` when a shorter reply length would (#1875); and
        `step.over_window` when the step cannot fit even as excerpts or the
        fitted request is still over. Without a known window there is nothing to fit
        against, and the per-result cap is the only bound.

        The step was logged as it entered, as ingested (#1443): cutting it to shares or
        refusing it takes results out of the history, never out of the session log. A
        result cut to its share is written through the session's door, so the cut text is
        logged too; no request has shown the step, so nothing is declared.
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
            return "step.over_window"
        # `total` counts the conversation as the request renders it, a repeated result as
        # its one-line back-reference (§5.8, Rule 2); what is taken out of it here must be
        # counted the same way, or a step holding a back-reference looks smaller outside
        # itself and larger inside than it is (#1854).
        rendered, refers = render_conversation(history)
        # Everything but the step itself -- its calls and its results -- already reaches
        # the window: no share of any size fits, and fewer calls would not help.
        outside = total - estimate_message_tokens(rendered[start:])
        if outside + reserve >= window:
            logger.warning(
                "No room is left in a %d-token window for a step's %d tool results: the "
                "request without the step is %d tokens and %d are kept for the reply",
                window,
                len(tail),
                outside,
                reserve,
            )
            # The system turn and the tool schemas are sent whatever the conversation
            # holds, so when they alone reach the window no shortening makes room (#1866).
            prepared = self._prepare_turn_messages(extra_sections=extra_sections)
            setup = tuple(
                m for m in prepared if m.role is MessageRole.SYSTEM and not m.compaction_ledger
            )
            fixed = estimate_request_tokens(LLMRequest(messages=setup, tools=tuple(tools)))
            if fixed + reserve >= window:
                # A reply length above the default is the other cause: when the setup
                # leaves room for a reply of the default size, a shorter one fits (#1875).
                if fixed + STEP_REPLY_RESERVE_TOKENS < window:
                    return "step.no_room_setup_reply"
                return "step.no_room_setup"
            if self._config.llm_config.auto_compact:
                return "step.no_room"
            return "step.no_room_no_compaction"
        # A result sent as a back-reference is not shared out: it costs its one line
        # whatever its size, and cutting it would send an excerpt in its place.
        shared_out = [i for i in tail if refers[i] is None]
        current = [history[i].content or "" for i in shared_out]
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
            return "step.over_window"
        raw_of = dict(zip(tail, step_results, strict=True))
        cut: dict[int, ChatMessage] = {}
        for index, content, cap in zip(shared_out, current, caps, strict=True):
            raw = raw_of[index]
            if raw.content is None or len(content.encode("utf-8")) <= cap:
                continue
            shared = self._ingest_tool_message(raw, readable=readable, cap_bytes=cap)
            # The form and what it was cut from are the new excerpt's, cap included: the
            # log keeps the form, not its text, so a `rendered_from` left at the ingest cap
            # would render the result uncut on the wire (#1848).
            cut[index] = history[index].model_copy(
                update={
                    "content": shared.content,
                    "form": shared.form or ContextForm.EXCERPT.value,
                    "rendered_from": shared.rendered_from,
                }
            )
            self._active_session.replace(index, cut[index], cause="step_cut")
        # A later result of the step that referred back to one just cut still carries the
        # text it was the same as; given the cut form too, it stays a back-reference.
        for index in tail:
            earlier = refers[index]
            if earlier is not None and earlier in cut:
                self._active_session.replace(
                    index,
                    history[index].model_copy(
                        update={
                            "content": cut[earlier].content,
                            "form": cut[earlier].form,
                            "rendered_from": cut[earlier].rendered_from,
                        }
                    ),
                    cause="step_cut",
                )
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
            return "step.over_window"
        return None

    def _withhold_refused_step(self, live: TurnSession) -> int:
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
        # Nothing here was shown, so the cut declares nothing (`truncate`).
        live.truncate(start, cause="step_refused")
        live.updated_at = _now_iso()
        logger.warning(
            "Withheld a refused step (%d tool calls) from session %s of agent %s",
            len(withheld),
            self._context.session_id,
            self.agent_id,
        )
        return len(withheld)

    def _evaluate_tool_guardrail(
        self,
        tool_calls: Sequence[ToolCallRequest],
        prior_executions: Sequence[ToolExecutionRecord],
        req: LLMRequest,
    ) -> str | None:
        """Check whether tool execution should be intercepted to prevent loops or context overflow."""
        called_names = {tc.name for tc in tool_calls}
        has_prior_image_gen = any(
            rec.tool_name == "generate_image" and rec.status == ToolResultStatus.SUCCESS
            for rec in prior_executions
        )
        if "generate_image" in called_names and has_prior_image_gen:
            return (
                "Images have already been generated in this turn. Further image generation in the same "
                "turn is restricted to prevent duplicate generation loops. Do not call any further tools; "
                "immediately conclude your final response displaying the images generated so far, each "
                "embedded by the relative_url its result returned."
            )

        window = self._context_window()
        reserve = self._reply_reserve()
        if window is not None and prior_executions:
            current_tokens = estimate_request_tokens(
                LLMRequest(messages=tuple(req.messages), tools=tuple(req.tools))
            )
            is_image_call = "generate_image" in called_names
            threshold = window - 512 if is_image_call else window
            if current_tokens + reserve >= threshold:
                return (
                    "Context token limit is approaching. Further tool calls in this turn are restricted "
                    "to prevent context overflow. Do not call any further tools; immediately conclude "
                    "your final response using the tool results obtained so far."
                )

        return None

    def _synthesize_guarded_tool_results(
        self,
        tool_calls: Sequence[ToolCallRequest],
        notice: str,
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord]]:
        """Synthesize skipped tool result messages to satisfy protocol and nudge model."""
        messages: list[ChatMessage] = []
        executions: list[ToolExecutionRecord] = []
        for tc in tool_calls:
            guard_payload: Any = {"status": "skipped", "notice": notice}
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=canonical_tool_text(guard_payload),
                    name=tc.name,
                    tool_call_id=tc.id,
                )
            )
            executions.append(
                ToolExecutionRecord(
                    tool_name=tc.name,
                    arguments=cast(dict[str, Any], unwrap_immutable(tc.arguments)),
                    output=guard_payload,
                    status=ToolResultStatus.SUCCESS,
                    error=None,
                    duration_ms=0.0,
                    tool_call_id=tc.id,
                )
            )
        return messages, executions

    async def _execute_step_tools(
        self,
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest],
        tool_executions: Sequence[ToolExecutionRecord],
        req: LLMRequest,
        tool_defs: list[ToolDefinition],
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None,
        shown_tool_names: set[str],
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord], bool]:
        """Execute or intercept tool calls, applying loop and budget guardrails."""
        guard_notice = self._evaluate_tool_guardrail(tool_calls, tool_executions, req)
        if guard_notice is not None:
            tool_messages, step_executions = self._synthesize_guarded_tool_results(
                tool_calls, guard_notice
            )
            tool_defs.clear()
            shown_tool_names.clear()
            return tool_messages, step_executions, True

        tool_messages, step_executions = await self._execute_tools(
            tuple(tool_calls),
            stream_callback=stream_callback,
            advertised=frozenset(shown_tool_names),
        )
        if self._tool_invoker.tools_module == "k_act":
            self._announce_bound_on_call(tool_messages, step_executions, shown_tool_names)
        return tool_messages, step_executions, False

    async def _model_accepts_images(
        self, tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest]
    ) -> bool:
        """Whether this turn's model takes image input, for a tool that can return one.

        Asked only when a called tool declares `returns_images`, so an ordinary step reads
        no listing. A provider that cannot say answers False, so a tool returns text
        rather than a picture the model would be sent and could not read (#2107).
        """
        llm = self._llm
        if not isinstance(llm, ImageInputProbe):
            return False
        if not any(tool_returns_images(self._tool_invoker.resolve(tc.name)) for tc in tool_calls):
            return False
        return await llm.accepts_images(self._config.llm_config.model_name or None)

    async def execute_tools(
        self,
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest],
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        advertised: frozenset[str] | None = None,
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord]]:
        """Execute a sequence of tool calls concurrently with P3 sandbox containment and path safety.

        `advertised` is the tool names the request that produced these calls declared.
        When given, a call to any other name is refused before it runs (F12); `None` is a
        direct call, which no request produced, and skips that check.
        """
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
            skill_dirs=self.active_skill_dirs(),
            isolation=isolation,
            turn_index=self._turn_counter,
            agent_delegate=self._agent_delegate,
            stored_results=self._result_bodies(self._context.session_id),
            room_id=self._turn_room_id,
            turn_id=self._turn_caller_turn_id,
            story_id=self._turn_story_id,
            person_names=self._turn_person_names,
            # The clone's own names, so `record_memory_fact` files a fact about it under
            # `self` (#2016). A name it shares with the person stays the person's.
            clone_names=tuple(n for n in (self.agent_id, self._config.name) if n),
            # The clone's own picture model, so a picture it asks for uses it (§3.4).
            image_model=self._config.llm_config.image_model,
            accepts_images=await self._model_accepts_images(tool_calls),
        )

        if len(tool_calls) == 1:
            msg, rec = await self._execute_single_tool(
                tool_calls[0], tool_ctx, stream_callback=stream_callback, advertised=advertised
            )
            self._after_tool_step((rec,), tool_ctx)
            return [msg], [rec]

        # Concurrently execute multiple tool calls (Issue #185)
        results = await asyncio.gather(
            *(
                self._execute_single_tool(
                    tc, tool_ctx, stream_callback=stream_callback, advertised=advertised
                )
                for tc in tool_calls
            )
        )
        tool_messages = [r[0] for r in results]
        tool_executions = [r[1] for r in results]
        self._after_tool_step(tool_executions, tool_ctx)
        return tool_messages, tool_executions
