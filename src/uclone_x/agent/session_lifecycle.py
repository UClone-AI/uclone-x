"""Session lifecycle: the sessions an agent hosts, and switching, loading and saving them.

Moved out of `agent/base.py` unchanged (#1736, stage 4). `BaseAgent` stays the facade:
`switch_session`, `load_history`, `get_session`, `checkpoint_turn`, `roll_back_turn`,
`reset_session`, `persist_session`, `hydrate_session` and `delete_session` are its public
methods and delegate here, and it keeps `_live_session`, `_seed_live_session`,
`_effective_session_id`, `_refuse_session_mutation_during_turn`, `_subscription_topics`
and `_write_pending_bodies` as delegators, because other modules and tests call them. The
moved code calls those back *through the agent*, so one replaced on the instance is the
one it runs.

The lifecycle holds no state of its own. The hosted sessions, the durable-event queue,
the per-session compactors and the stranded-event counts stay on the agent, and are read
and written through a `SessionScope` of callables evaluated on every access, so an agent
whose `_store`, `_sessions` or `_context` is swapped after construction is read as it now
is. The accessors carry the names the agent's attributes have, so the moved code reads as
it did on `BaseAgent`.

`_LiveSession`, the mutable working copy of one session, moved here with it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from uclone_x.agent.models import AgentConfig, AgentContext, PersonaDefinition, PlanState
from uclone_x.agent.prompt_assembler import (
    AnchorWriter,
    LiveAnchorProvenance,
    has_anchor,
    persisted_anchor_provenance,
    restored_anchor_provenance,
)
from uclone_x.agent.session import (
    ContextSnapshot,
    SessionState,
    cleanup_session_artifacts,
    redact_message,
    validate_session_id,
)
from uclone_x.agent.tool_invoker import ToolInvoker
from uclone_x.core.context_state import EPOCH_RESTORED, ContextEntry, ContextEpoch, advance
from uclone_x.core.session_log import (
    LoggedMessage,
    SessionLogEntry,
    SessionLogProvenance,
    history_entry_ids,
    logged_message,
    new_entry,
)
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import artifacts_dir_for
from uclone_x.engine.event_bus import UnauthorizedSubscriptionError
from uclone_x.engine.protocols import EventSubscriptionProtocol
from uclone_x.errors import (
    SessionMutationDuringTurnError,
    SessionStoreNotConfiguredError,
    SessionSwitchWhileRunningError,
)
from uclone_x.llm.models import ChatMessage, ToolCallRequest
from uclone_x.llm.protocols import ContextCompactorProtocol

__all__ = [
    "SessionLifecycle",
    "SessionScope",
]

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


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
    #: it through `persisted_anchor_provenance` and `hydrate_session` reads it back through
    #: `restored_anchor_provenance`, so a restored session says which persona composed its
    #: anchor instead of attributing every restored anchor to the caller. A record that
    #: predates the field comes back `UNRECORDED` — reported, and still not re-resolved.
    anchor_provenance: LiveAnchorProvenance = AnchorWriter.CALLER
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
    #: The memory section recalled for the running turn's message (clone-knowledge-graph
    #: §3.5), set once before the turn's first request and sent on each of its steps, so
    #: the turn-context tail holds still within the turn. `None` until a turn with a
    #: memory store starts; not persisted, since the next turn recomputes it.
    recalled_memory: str | None = None
    #: Catalog tools host binding and `search_tools` have appended to this session's tools
    #: layer, in the order they were bound (design §5.1). Grow-only: cleared only at
    #: compaction, where the request prefix is rebuilt anyway. Not persisted as such: a
    #: restored session is reseeded from the catalog tools its history called
    #: (`ToolInvoker.reseed_bound_tools_from_history`), so a follow-up that repeats one of
    #: those calls after a restart still finds the tool declared (#1545 item 3).
    bound_tools: list[str] = field(default_factory=list[str])
    #: Set when binding failed for this session: every held tool is pinned from then until
    #: compaction. Pinning is a superset of any bound set, so falling back only grows the
    #: layer, and it never flips back, which would remove tools mid-session.
    tools_pin_all: bool = False
    #: Every message that entered `messages`, append-only (#1443). Persisted as
    #: `SessionState.session_log`; each entry's body waits in `pending_bodies` like a
    #: snapshot layer's does. Written only by `log_history`.
    session_log: list[SessionLogEntry] = field(default_factory=list[SessionLogEntry])
    #: Each message now in `messages`, as of the last `log_history`: the message, its
    #: rendering for the log, and the log entry it is. The message is held, not its
    #: `id()`, so an id reused after collection never matches. `None` until the first
    #: `log_history`, which then matches the history to the log by digest
    #: (`history_entry_ids`). Not persisted: the record's log and messages rebuild it.
    aligned: list[tuple[ChatMessage, LoggedMessage, str]] | None = None
    #: What each request of this session showed, per epoch (#1443, design §5.8).
    #: Persisted as `SessionState.context_epochs`. Written only by `record_shown`.
    context_epochs: list[ContextEpoch] = field(default_factory=list[ContextEpoch])
    #: Why the next request may not extend the current epoch, as declared since the last
    #: request: a compaction, a rollback, a retry, a replaced or restored history. The
    #: next `record_shown` that opens an epoch records them, and every request clears them.
    #: Persisted as `SessionState.epoch_causes`, all but `restored`, so an epoch a
    #: compaction opens is still named for it after a restart (#1848).
    epoch_causes: list[str] = field(default_factory=list[str])
    #: Log bodies already decoded back into messages, by digest (`logged_history`). Not
    #: persisted: a cache of `ChatMessage.model_validate_json` over bodies the log holds.
    decoded: dict[str, ChatMessage] = field(default_factory=dict[str, ChatMessage])
    #: Per log entry of a compacted history, the entry it shows and its form, derived at
    #: the compaction (`compacted_entries`, #1848), for the request that opens the new
    #: epoch. A pruned message is the rendering of the entry it replaced. Set by the
    #: compaction driver and cleared by `record_shown`. Persisted as
    #: `SessionState.compacted_entries`, keyed here by `ContextEntry.body`, so a restart
    #: before that request shows the same entries and renderings.
    compacted_entries: dict[str, ContextEntry] = field(default_factory=dict[str, ContextEntry])

    @classmethod
    def from_state(
        cls,
        state: SessionState,
        *,
        anchor_provenance: LiveAnchorProvenance,
        log_as: SessionLogProvenance = SessionLogProvenance.RECORDED,
    ) -> _LiveSession:
        """Adopt a persisted or seeded session as the live working copy.

        `anchor_provenance` is keyword-only and has no default here, so a call that omits
        it does not type-check: deciding who composed the anchor is part of putting a
        session into `_sessions`, and a default would let a new call inherit that answer
        by omission. The field's own default exists for the dataclass, not for callers.

        `log_as` defaults to `RECORDED`; only `hydrate_session` passes `MIGRATED`. The
        record's log is reconciled with its messages: a message the log does not account for gets an entry, `MIGRATED`
        with no turn when the record was read from a store (it predates the log), or
        `RECORDED` at the record's turn when a caller just supplied it. A record whose log
        already accounts for every message gains nothing, so loading twice adds nothing.
        """
        live = cls(
            messages=list(state.messages),
            plan=state.plan,
            turn_counter=state.turn_counter,
            created_at=state.created_at,
            updated_at=state.updated_at,
            revision=state.revision,
            anchor_provenance=anchor_provenance,
            context_snapshots=list(state.context_snapshots),
            session_log=list(state.session_log),
            context_epochs=list(state.context_epochs),
            compacted_entries={shown.body: shown for shown in state.compacted_entries},
            epoch_causes=list(state.epoch_causes),
        )
        live.log_history(provenance=log_as)
        return live

    def log_history(
        self, *, provenance: SessionLogProvenance = SessionLogProvenance.RECORDED
    ) -> None:
        """Log every message in `messages` the log does not yet account for (#1443).

        The one writer of `session_log`. Called where a turn adds to the history, before a
        rollback or a replacement removes from it, and by `to_state` and every save, so a
        message is logged even if it leaves the history before the turn ends and no record
        names an entry whose body is not queued for writing.

        Each message is matched to an entry: first a message object that was already in
        the history keeps its entry, then a message whose digest an entry of the previous
        history carried and no message kept takes that entry, in order. What is left is
        logged. So accounting is a multiset by digest against what the history held at the
        last call: a message that stayed is not logged again, and one that left and came
        back -- a retried prompt after a rollback -- is logged again, because it entered
        again. A `RECORDED` entry carries the session's turn counter; a `MIGRATED` one, none.
        """
        rendering: list[LoggedMessage] = []
        claimed: list[str | None] = [None] * len(self.messages)
        taken: set[str] = set()
        by_object = {id(m): (m, item, entry) for m, item, entry in self.aligned or ()}
        for index, message in enumerate(self.messages):
            known = by_object.get(id(message))
            if known is not None and known[0] is message and known[2] not in taken:
                claimed[index] = known[2]
                taken.add(known[2])
                rendering.append(known[1])
            else:
                rendering.append(logged_message(redact_message(message)))
        if self.aligned is None:
            pool_ids = history_entry_ids(self.session_log, [item.digest for item in rendering])
            previous = [
                (entry, item.digest)
                for entry, item in zip(pool_ids, rendering, strict=True)
                if entry is not None
            ]
        else:
            previous = [(entry, item.digest) for _m, item, entry in self.aligned]
        pools: dict[str, list[str]] = {}
        for entry, digest in reversed(previous):
            if entry not in taken:
                pools.setdefault(digest, []).append(entry)
        aligned: list[tuple[ChatMessage, LoggedMessage, str]] = []
        for index, (message, item) in enumerate(zip(self.messages, rendering, strict=True)):
            entry = claimed[index]
            if entry is None and pools.get(item.digest):
                entry = pools[item.digest].pop()
            if entry is None:
                logged = new_entry(
                    len(self.session_log),
                    item,
                    turn=None if provenance is SessionLogProvenance.MIGRATED else self.turn_counter,
                    provenance=provenance,
                )
                self.session_log.append(logged)
                entry = logged.id
                if item.digest not in self.stored_bodies:
                    self.pending_bodies[item.digest] = item.body
            aligned.append((message, item, entry))
        self.aligned = aligned

    def log_entry(self, rendered: LoggedMessage) -> SessionLogEntry:
        """Log something a request sent that is not a history message (#1849).

        The recalled memory section of a turn is one: it is sent in the `[Turn Context]`
        tail, never in `messages`, so `log_history` does not see it. The history is logged
        first, so the entry follows the message it was recalled for. Nothing a request
        sends is changed: the entry is only a record, and its body waits in
        `pending_bodies` like any other.
        """
        self.log_history()
        logged = new_entry(
            len(self.session_log),
            rendered,
            turn=self.turn_counter,
            provenance=SessionLogProvenance.RECORDED,
        )
        self.session_log.append(logged)
        if rendered.digest not in self.stored_bodies:
            self.pending_bodies[rendered.digest] = rendered.body
        return logged

    def entry_ids(self) -> list[str]:
        """The log entry each message in `messages` is, logging any that is not yet."""
        self.log_history()
        assert self.aligned is not None
        return [entry for _message, _item, entry in self.aligned]

    def logged_history(self) -> list[tuple[str, ChatMessage]]:
        """Each message in `messages` as the session log holds it: its entry, and the
        message decoded from the entry's body (#1848).

        `messages` names which entries are in the history and in what order; the text a
        request shows is read from the log.
        """
        self.log_history()
        assert self.aligned is not None
        logged: list[tuple[str, ChatMessage]] = []
        for _message, item, entry in self.aligned:
            message = self.decoded.get(item.digest)
            if message is None:
                message = ChatMessage.model_validate_json(item.body)
                self.decoded[item.digest] = message
            logged.append((entry, message))
        return logged

    def shown_in_epoch(self, index: int) -> bool:
        """Whether the message at `index` of `messages` is an entry the current epoch shows.

        A message no request of this epoch has shown -- a final answer the model just
        returned -- can still be rewritten without breaking Rule 1: the next request only
        appends it, so it does not declare a new epoch (#1854).
        """
        if not self.context_epochs or not 0 <= index < len(self.messages):
            return False
        entry = self.entry_ids()[index]
        return any(shown.body == entry for shown in self.context_epochs[-1].entries)

    def declare_new_epoch(self, cause: str) -> None:
        """Say that the next request may show the history differently, and why (Rule 1)."""
        if cause not in self.epoch_causes:
            self.epoch_causes.append(cause)

    def record_shown(self, shown: list[ContextEntry], *, step: int) -> ContextEpoch:
        """Record what a request's conversation showed; returns the epoch it belongs to.

        The epoch is extended when the request only appended to it, and a new one opens
        otherwise, naming the causes declared since the last request (`declare_new_epoch`).
        """
        self.context_epochs = list(
            advance(
                self.context_epochs,
                shown,
                turn=self.turn_counter,
                step=step,
                opened_by=self.epoch_causes,
            )
        )
        self.epoch_causes = []
        self.compacted_entries = {}
        return self.context_epochs[-1]

    def to_state(self, session_id: str, agent_id: str) -> SessionState:
        """Snapshot this session into the frozen shape the store persists.

        **This is the only door that writes `anchor_provenance` onto a record**, because
        it is the only place the answer is held. Every route to `SessionStore.save` that
        an agent takes passes through here, so stamping it here rather than at each save
        is what keeps the record's account of its anchor and the anchor itself together.
        """
        self.log_history()
        return SessionState(
            session_id=session_id,
            agent_id=agent_id,
            messages=tuple(self.messages),
            plan=self.plan,
            turn_counter=self.turn_counter,
            created_at=self.created_at,
            updated_at=self.updated_at,
            revision=self.revision,
            anchor_provenance=persisted_anchor_provenance(self.anchor_provenance),
            context_snapshots=tuple(self.context_snapshots),
            session_log=tuple(self.session_log),
            context_epochs=tuple(self.context_epochs),
            compacted_entries=tuple(self.compacted_entries.values()),
            # `restored` is declared again by every load, so it is not saved (#1848).
            epoch_causes=tuple(c for c in self.epoch_causes if c != EPOCH_RESTORED),
        )


@dataclass(frozen=True, slots=True)
class SessionScope:
    """What a `SessionLifecycle` reads and writes on the agent it serves, each read when needed.

    Callables rather than values: a switch replaces the agent's context, a reset rebinds
    its durable-event queue, and a caller may swap the store or the session map after
    construction. The agent's own methods are reached through getters that return the
    agent's *current* bound method, so one a test or a subclass replaces on the instance
    is the one the lifecycle calls.
    """

    #: The agent's id.
    agent_id: Callable[[], str]
    #: The agent's `AgentConfig`.
    config: Callable[[], AgentConfig]
    #: The agent's `AgentContext`; its `session_id` names the active session.
    context: Callable[[], AgentContext]
    set_context: Callable[[AgentContext], None]
    #: The Core session store, or `None` when none is wired.
    store: Callable[[], SessionStoreProtocol | None]
    #: The hosted sessions by id, the dict itself (it is written in place).
    sessions: Callable[[], dict[str, _LiveSession]]
    #: The lock that serializes the agent's turns.
    turn_lock: Callable[[], asyncio.Lock]
    #: Whether the agent's event loop is running.
    running: Callable[[], bool]
    #: The agent's bus subscription, or `None`.
    subscription: Callable[[], EventSubscriptionProtocol | None]
    #: Events a switch discarded, by session, the dict itself.
    stranded_event_counts: Callable[[], dict[str, int]]
    #: The agent's queue of durable events awaiting the store.
    pending_durable_events: Callable[[], list[dict[str, Any]]]
    set_pending_durable_events: Callable[[list[dict[str, Any]]], None]
    #: The per-session compactors, the dict itself.
    session_compactors: Callable[[], dict[str, ContextCompactorProtocol]]
    #: The steps taken in the current run.
    run_steps: Callable[[], int]
    set_run_steps: Callable[[int], None]
    #: The names of the skills this agent has loaded, the set itself.
    loaded_skills: Callable[[], set[str]]
    #: The agent's tool invoker.
    tool_invoker: Callable[[], ToolInvoker]
    #: The system prompt a seeded or reset session is anchored with.
    effective_system_prompt: Callable[[], str]
    # -- the agent's methods, as getters of the current bound method --------------
    resolved_persona: Callable[[], Callable[[], PersonaDefinition | None]]
    resolve_workspace_root: Callable[[], Callable[[], Path | None]]
    effective_session_id: Callable[[], Callable[[str | None], str]]
    refuse_session_mutation_during_turn: Callable[[], Callable[[str, str], None]]
    seed_live_session: Callable[[], Callable[[str], _LiveSession]]
    live_session: Callable[[], Callable[[str], _LiveSession]]
    get_session: Callable[[], Callable[[str | None], SessionState]]
    subscription_topics: Callable[[], Callable[[str], set[str]]]
    write_pending_bodies: Callable[[], Callable[[str], None]]


class SessionLifecycle:
    """Hosts one agent's sessions: seeds, switches, loads, resets, saves and restores them."""

    def __init__(self, scope: SessionScope) -> None:
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

    @_context.setter
    def _context(self, value: AgentContext) -> None:
        self._scope.set_context(value)

    @property
    def _store(self) -> SessionStoreProtocol | None:
        return self._scope.store()

    @property
    def _sessions(self) -> dict[str, _LiveSession]:
        return self._scope.sessions()

    @property
    def _turn_lock(self) -> asyncio.Lock:
        return self._scope.turn_lock()

    @property
    def _running(self) -> bool:
        return self._scope.running()

    @property
    def _subscription(self) -> EventSubscriptionProtocol | None:
        return self._scope.subscription()

    @property
    def _stranded_event_counts(self) -> dict[str, int]:
        return self._scope.stranded_event_counts()

    @property
    def _pending_durable_events(self) -> list[dict[str, Any]]:
        return self._scope.pending_durable_events()

    @_pending_durable_events.setter
    def _pending_durable_events(self, value: list[dict[str, Any]]) -> None:
        self._scope.set_pending_durable_events(value)

    @property
    def _session_compactors(self) -> dict[str, ContextCompactorProtocol]:
        return self._scope.session_compactors()

    @property
    def _run_steps(self) -> int:
        return self._scope.run_steps()

    @_run_steps.setter
    def _run_steps(self, value: int) -> None:
        self._scope.set_run_steps(value)

    @property
    def _loaded_skills(self) -> set[str]:
        return self._scope.loaded_skills()

    @property
    def _tool_invoker(self) -> ToolInvoker:
        return self._scope.tool_invoker()

    @property
    def effective_system_prompt(self) -> str:
        return self._scope.effective_system_prompt()

    # -- the agent's methods, called back through it -------------------------------

    @property
    def _resolved_persona(self) -> Callable[[], PersonaDefinition | None]:
        return self._scope.resolved_persona()

    @property
    def _resolve_workspace_root(self) -> Callable[[], Path | None]:
        return self._scope.resolve_workspace_root()

    @property
    def _effective_session_id(self) -> Callable[[str | None], str]:
        return self._scope.effective_session_id()

    @property
    def _refuse_session_mutation_during_turn(self) -> Callable[[str, str], None]:
        return self._scope.refuse_session_mutation_during_turn()

    @property
    def _seed_live_session(self) -> Callable[[str], _LiveSession]:
        return self._scope.seed_live_session()

    @property
    def _live_session(self) -> Callable[[str], _LiveSession]:
        return self._scope.live_session()

    @property
    def _get_session(self) -> Callable[[str | None], SessionState]:
        return self._scope.get_session()

    @property
    def _subscription_topics(self) -> Callable[[str], set[str]]:
        return self._scope.subscription_topics()

    @property
    def _write_pending_bodies(self) -> Callable[[str], None]:
        return self._scope.write_pending_bodies()

    # -- the lifecycle -------------------------------------------------------------

    def effective_session_id(self, session_id: str | None) -> str:
        """Resolve an optional session argument to a concrete id.

        `None` means "the active session". An empty string does **not**: the `or` idiom
        this replaces treated `""` as absent, so `persist_session("")` silently wrote the
        *active* session under the active id, and `reset_session("")` reset a session the
        caller had not named. That is the empty-id ambiguity `validate_session_id`
        refuses one layer down, and it should not be reintroduced by an idiom here.
        """
        return self._context.session_id if session_id is None else session_id

    def refuse_session_mutation_during_turn(self, session_id: str, operation: str) -> None:
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

    def seed_live_session(self, session_id: str) -> _LiveSession:
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
            log_as=SessionLogProvenance.RECORDED,
        )

    def live_session(self, session_id: str) -> _LiveSession:
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

    def subscription_topics(self, session_id: str) -> set[str]:
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
        replaced = _LiveSession.from_state(
            state, anchor_provenance=AnchorWriter.CALLER, log_as=SessionLogProvenance.RECORDED
        )
        # The snapshots and the log carry over, and so must the bodies they name that are
        # not written yet: dropping them left the next save naming bodies never stored.
        replaced.pending_bodies = {**live.pending_bodies, **replaced.pending_bodies}
        replaced.stored_bodies = live.stored_bodies
        replaced.declare_new_epoch("history_replaced")
        self._tool_invoker.reseed_bound_tools_from_history(replaced)
        self._sessions[sid] = replaced

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
          (`undone_attempt_section`), so a retry can see that a call already ran (#1495).

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
            # Logged first: the undone turn's messages leave the history, not the log.
            live.log_history()
            live.messages = list(checkpoint.messages)
            live.log_history()
            live.declare_new_epoch("rollback")
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
        reset = self._get_session(sid).reset(
            system_prompt=self.effective_system_prompt,
            anchor_provenance=persisted_anchor_provenance(self._resolved_persona()),
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
            self._get_session(sid), pending_events=events_to_persist if events_to_persist else None
        )
        if pending_events is None:
            self._pending_durable_events = [
                d_event
                for d_event in self._pending_durable_events
                if d_event.get("session_id") != sid  # persist_session retains other session events
            ]
        self._live_session(sid).revision = saved.revision
        return saved

    def write_pending_bodies(self, sid: str) -> None:
        """Write the bodies `sid`'s snapshots name and the store does not hold yet.

        Called before every save of a working copy, so a record that names a body never
        reaches the disk without it.
        """
        if self._store is None:
            return
        live = self._live_session(sid)
        # Log first, so the log entries `to_state` is about to record have their bodies
        # queued here rather than after this write (#1443).
        live.log_history()
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
        restored = restored_anchor_provenance(loaded.anchor_provenance)
        if restored is AnchorWriter.UNRECORDED and has_anchor(loaded.messages):
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
        # A record from before the session log backfills one `MIGRATED` entry per message;
        # one written since accounts for every message already, so this adds nothing.
        hydrated = _LiveSession.from_state(
            loaded, anchor_provenance=restored, log_as=SessionLogProvenance.MIGRATED
        )
        hydrated.declare_new_epoch(EPOCH_RESTORED)
        self._tool_invoker.reseed_bound_tools_from_history(hydrated)
        self._sessions[sid] = hydrated
        return loaded

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
