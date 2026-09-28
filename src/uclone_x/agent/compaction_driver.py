"""Context compaction: when a session is compacted, and committing and announcing it (P5).

Moved out of `agent/base.py` unchanged (#1736, stage 4). `BaseAgent` stays the facade:
`compact_session` is its public method and delegates here, and it keeps
`_session_compactor`, `_compact_session`, `_should_compact_session` and
`_auto_compact_if_needed` as delegators, because the turn loop and tests call them and
tests patch them on the instance. The moved code calls those back *through the agent*,
so a patched one is the one a compaction runs.

The driver holds no state of its own. The per-session compactors, the injected
compactor, the live sessions and the publisher stay on the agent and are read through a
`CompactionScope` of callables evaluated on every access. The accessors carry the names
the agent's attributes have, so the moved code reads as it did on `BaseAgent`.

Compaction is #183 requirement 3, under P5.
The P5 split this implements: the LLM layer owns the compaction algorithm and the
token estimate; the Core owns only persisting the compacted sequence into session
state and publishing the notice, because it is the component holding the publisher
and the store. `p5-llm-token-management.md` requires compaction to be "strictly
managed by the LLM layer, completely decoupled from agent business logic", while
`event-driven-agent-core.md` places pruning in the agent's INGESTING state and this
issue's requirement 3 says to wire it into `execute_turn`. Nothing here re-implements
a compaction decision or a token count; both are read off the compactor.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.agent.session import CompactionResult
from uclone_x.agent.session_lifecycle import _LiveSession  # pyright: ignore[reportPrivateUsage]
from uclone_x.agent.tool_invoker import ToolInvoker
from uclone_x.core.context_state import (
    CompactedForms,
    ContextEntry,
    compacted_entries,
    derive_compacted_forms,
    recorded_forms,
    recorded_renderings,
    shown_entries,
)
from uclone_x.core.session_log import LoggedMessage
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import TOOL_RESULT_READ_TOOL
from uclone_x.engine.event_bus import AgentEvent, EventPriority, EventType
from uclone_x.engine.protocols import PublisherHandleProtocol
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.models import ChatMessage, LLMRequest, ToolDefinition
from uclone_x.llm.protocols import ContextCompactorProtocol, TokenBudgetManagerProtocol
from uclone_x.telemetry.protocols import TracerProtocol
from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = [
    "CompactionDriver",
    "CompactionScope",
]

logger = logging.getLogger(__name__)


class _CompactSession(Protocol):
    def __call__(
        self,
        sid: str,
        reason: str,
        *,
        reader_offered: bool | None = ...,
    ) -> Awaitable[CompactionResult]: ...


class _ShouldCompactSession(Protocol):
    def __call__(
        self,
        session_id: str,
        messages: Sequence[ChatMessage],
        *,
        request: LLMRequest | None = ...,
    ) -> bool: ...


class _PrepareTurnMessages(Protocol):
    def __call__(self, extra_sections: Sequence[str] = ...) -> list[ChatMessage]: ...


@dataclass(frozen=True, slots=True)
class CompactionScope:
    """What a `CompactionDriver` reads on the agent it serves, each read when needed.

    Callables rather than values, because a compaction runs against the agent as it is
    now: its store, registry and budget can be swapped after construction, and the active
    session moves with every switch. The agent's own methods are reached through getters
    that return the agent's *current* bound method, so one a test or a subclass replaces
    on the instance is the one the driver calls.
    """

    #: The agent's id.
    agent_id: Callable[[], str]
    #: The agent's `AgentConfig`.
    config: Callable[[], AgentConfig]
    #: The agent's `AgentContext`; its `session_id` names the active session.
    context: Callable[[], AgentContext]
    #: The agent's tool registry, or `None`.
    tool_registry: Callable[[], ToolRegistryProtocol | None]
    #: The Core session store, or `None`.
    store: Callable[[], SessionStoreProtocol | None]
    #: The token budget manager, or `None`.
    budget: Callable[[], TokenBudgetManagerProtocol | None]
    #: The agent's bus publisher handle, or `None` without a bus.
    publisher: Callable[[], PublisherHandleProtocol | None]
    #: The agent's tracer.
    tracer: Callable[[], TracerProtocol]
    #: The agent's record of absorbed failures, the deque itself.
    processing_errors: Callable[[], deque[BaseException]]
    #: The agent's tool invoker.
    tool_invoker: Callable[[], ToolInvoker]
    #: The compactor the agent was given, shared by every session, or `None`.
    injected_compactor: Callable[[], ContextCompactorProtocol | None]
    #: The per-session compactors, the dict itself.
    session_compactors: Callable[[], dict[str, ContextCompactorProtocol]]
    #: The active session's history.
    history: Callable[[], list[ChatMessage]]
    # -- the agent's methods, as getters of the current bound method --------------
    live_session: Callable[[], Callable[[str], _LiveSession]]
    effective_session_id: Callable[[], Callable[[str | None], str]]
    refuse_session_mutation_during_turn: Callable[[], Callable[[str, str], None]]
    write_pending_bodies: Callable[[], Callable[[str], None]]
    resolve_workspace_root: Callable[[], Callable[[], Path | None]]
    context_window: Callable[[], Callable[[], int | None]]
    explicit_threshold: Callable[[], Callable[[int], int | None]]
    observe_context_window: Callable[[], Callable[[], Awaitable[None]]]
    prepare_turn_messages: Callable[[], _PrepareTurnMessages]
    session_compactor: Callable[[], Callable[[str], ContextCompactorProtocol]]
    compact_session: Callable[[], _CompactSession]
    should_compact_session: Callable[[], _ShouldCompactSession]


class CompactionDriver:
    """Decides when one agent's sessions are compacted, and commits and announces each pass."""

    def __init__(self, scope: CompactionScope) -> None:
        self._scope = scope

    # -- the agent's state, read through the scope ---------------------------------

    @property
    def agent_id(self) -> str:
        return self._scope.agent_id()

    @property
    def _config(self) -> AgentConfig:
        return self._scope.config()

    @property
    def _context(self) -> AgentContext:
        return self._scope.context()

    @property
    def _tools(self) -> ToolRegistryProtocol | None:
        return self._scope.tool_registry()

    @property
    def _store(self) -> SessionStoreProtocol | None:
        return self._scope.store()

    @property
    def _budget(self) -> TokenBudgetManagerProtocol | None:
        return self._scope.budget()

    @property
    def _publisher(self) -> PublisherHandleProtocol | None:
        return self._scope.publisher()

    @property
    def _tracer(self) -> TracerProtocol:
        return self._scope.tracer()

    @property
    def _processing_errors(self) -> deque[BaseException]:
        return self._scope.processing_errors()

    @property
    def _tool_invoker(self) -> ToolInvoker:
        return self._scope.tool_invoker()

    @property
    def _injected_compactor(self) -> ContextCompactorProtocol | None:
        return self._scope.injected_compactor()

    @property
    def _session_compactors(self) -> dict[str, ContextCompactorProtocol]:
        return self._scope.session_compactors()

    @property
    def _history(self) -> list[ChatMessage]:
        return self._scope.history()

    # -- the agent's methods, called back through it -------------------------------

    @property
    def _live_session(self) -> Callable[[str], _LiveSession]:
        return self._scope.live_session()

    @property
    def _effective_session_id(self) -> Callable[[str | None], str]:
        return self._scope.effective_session_id()

    @property
    def _refuse_session_mutation_during_turn(self) -> Callable[[str, str], None]:
        return self._scope.refuse_session_mutation_during_turn()

    @property
    def _write_pending_bodies(self) -> Callable[[str], None]:
        return self._scope.write_pending_bodies()

    @property
    def _resolve_workspace_root(self) -> Callable[[], Path | None]:
        return self._scope.resolve_workspace_root()

    @property
    def _context_window(self) -> Callable[[], int | None]:
        return self._scope.context_window()

    @property
    def _explicit_threshold(self) -> Callable[[int], int | None]:
        return self._scope.explicit_threshold()

    @property
    def _observe_context_window(self) -> Callable[[], Awaitable[None]]:
        return self._scope.observe_context_window()

    @property
    def _prepare_turn_messages(self) -> _PrepareTurnMessages:
        return self._scope.prepare_turn_messages()

    @property
    def _session_compactor(self) -> Callable[[str], ContextCompactorProtocol]:
        return self._scope.session_compactor()

    @property
    def _compact_session(self) -> _CompactSession:
        return self._scope.compact_session()

    @property
    def _should_compact_session(self) -> _ShouldCompactSession:
        return self._scope.should_compact_session()

    # -- compaction ----------------------------------------------------------------

    def session_compactor(self, session_id: str) -> ContextCompactorProtocol:
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

    async def compact_session_unguarded(
        self,
        sid: str,
        reason: str,
        *,
        reader_offered: bool | None = None,
    ) -> CompactionResult:
        """Compact one session unconditionally, without the in-flight-turn guard.

        Split from `compact_session` because automatic compaction runs *inside*
        `execute_turn`, while `_turn_lock` is held — so the public method's guard would
        refuse the one caller that is allowed to compact inside a turn. That caller is
        safe for the reason the guard exists to protect: it compacts at turn start,
        *before* the turn's first request is built, so no in-flight assistant message can
        be orphaned, and it is the turn itself rather than a concurrent caller racing it.
        It never runs between two steps of a turn: an epoch starts only at a turn boundary
        (§5.8, owner ruling 2026-09-27, reversing #1422's pass between steps).

        `reader_offered` says whether this turn offers `tool_result_read`; the short form
        of a stored result names it, so it is used only when the model can call it.
        `None` asks what the agent holds -- the registry filtered by `allowed_tools`.
        """
        live = self._live_session(sid)
        compactor = self._session_compactor(sid)
        if isinstance(compactor, ContextCompactor):
            if reader_offered is None:
                # What the agent holds, not what is registered: an operator's
                # `allowed_tools` can leave the reader out of a registry that has it (#1653).
                reader_offered = any(
                    t.name == TOOL_RESULT_READ_TOOL for t in self._tool_invoker.held_tools()
                )
            compactor.tool_result_reader = reader_offered

        before_messages = list(live.messages)
        tokens_before = compactor.estimate_tokens(before_messages)

        outcome = await compactor.compact(before_messages)
        compacted = tuple(outcome.messages)
        derived = _derive_forms(live, before_messages, compacted, outcome.origins)
        if derived is not None and derived.rising:
            # A pruned message whose form would rank above the one it replaces is not
            # taken: the message it replaces stays, in the form it was shown in (Rule 3).
            logger.warning(
                "compaction kept %d message(s) whose pruned form would rank above the form "
                "they were shown in",
                len(derived.rising),
            )
            compacted = tuple(
                before_messages[origin]
                if position in derived.rising and origin is not None
                else message
                for position, (message, origin) in enumerate(
                    zip(compacted, outcome.origins, strict=True)
                )
            )
        tokens_after = compactor.estimate_tokens(compacted)

        # A failed write leaves the session untouched on both sides, never compacted in
        # memory and whole on disk. An earlier order mutated `live.messages` before the
        # save, so one failed `save` left the process believing a 25-message session was
        # 6 messages long while the record still held 25 — and the exception carried no
        # hint that the in-memory sequence had already been discarded. Compaction is
        # destructive and unrecoverable, so it commits atomically or not at all.
        #
        # The record it writes logs the compacted history and holds the entries derived
        # for it (#1848), so a restart before the next request reads each pruned message
        # as the entry it replaced, under the same ids, instead of logging it as an entry
        # of its own. Those are computed on the working copy, which is put back if the
        # write fails (`_Undo`).
        live.log_history()
        undo = _Undo.of(live)
        try:
            # `entry_ids` and `to_state` log the compacted history: the summary enters the
            # log at the turn it was made in, and the log's count of what the history
            # holds drops the folded turns, so a message identical to one of them that
            # arrives later is logged when it enters (#1443).
            live.messages = list(compacted)
            # The new epoch's entries and forms, derived from the entries the history
            # showed before the compaction: a pruned message is the entry it replaced, in
            # a smaller form, and its logged body is that form's rendering. The next
            # request shows them.
            live.compacted_entries = (
                {} if derived is None else compacted_entries(live.entry_ids(), derived)
            )
            # The one point a request may show the history in smaller forms: the next
            # request opens a new epoch (§5.8, Rule 3). Declared before the record is
            # built, so a restart before that request still names the compaction.
            live.declare_new_epoch("compaction")
            new_state = live.to_state(sid, self._config.agent_id)
            if self._store is not None:
                self._write_pending_bodies(sid)  # before the record that names them
                new_state = self._store.save(new_state)
        except BaseException:
            undo.restore(live)
            raise
        live.messages = list(new_state.messages)
        # The one point the tools layer may shrink (design §5.1): binding restarts from the
        # base set and a pinned session retries.
        live.bound_tools.clear()
        live.tools_pin_all = False
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

    def should_compact_session(
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

    async def auto_compact_if_needed(
        self,
        tools: Sequence[ToolDefinition] = (),
        extra_sections: Sequence[str] = (),
        *,
        reason: str = "auto_threshold",
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
            reader_offered=any(d.name == TOOL_RESULT_READ_TOOL for d in tools),
        )


@dataclass(frozen=True, slots=True)
class _Undo:
    """What a compaction changes on the working copy before its write, to put back if the
    write fails. The log only grows, so it is put back by length."""

    messages: list[ChatMessage]
    log_length: int
    aligned: list[tuple[ChatMessage, LoggedMessage, str]] | None
    pending_bodies: dict[str, str]
    compacted_entries: dict[str, ContextEntry]
    epoch_causes: list[str]

    @classmethod
    def of(cls, live: _LiveSession) -> _Undo:
        return cls(
            messages=list(live.messages),
            log_length=len(live.session_log),
            aligned=live.aligned,
            pending_bodies=dict(live.pending_bodies),
            compacted_entries=live.compacted_entries,
            epoch_causes=list(live.epoch_causes),
        )

    def restore(self, live: _LiveSession) -> None:
        live.messages = self.messages
        del live.session_log[self.log_length :]
        live.aligned = self.aligned
        live.pending_bodies = self.pending_bodies
        live.compacted_entries = self.compacted_entries
        live.epoch_causes = self.epoch_causes


def _derive_forms(
    live: _LiveSession,
    before: Sequence[ChatMessage],
    after: Sequence[ChatMessage],
    origins: Sequence[int | None],
) -> CompactedForms | None:
    """The entries and forms a compaction's result shows (`derive_compacted_forms`, #1848).

    Derived from what the history before it showed -- each message's entry and form, read
    from the log and the epochs as a request reads them (`shown_entries`) -- not from the
    messages the compactor wrote. `None` when the compactor did not say where its messages
    came from (an injected compactor may not), or said it inconsistently: the next request
    then reads each message as its own entry, as it does for any history (`shown_form`).
    """
    if not origins:
        return None
    epochs = live.context_epochs
    showing = shown_entries(
        live.logged_history(),
        recorded_forms(epochs),
        {**recorded_renderings(epochs), **live.compacted_entries},
    )
    try:
        return derive_compacted_forms(list(zip(showing, before, strict=True)), after, origins)
    except ValueError as exc:
        logger.warning("compaction forms not derived: %s", exc)
        return None
