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
from typing import Any

from uclone_x.agent.models import AgentConfig, AgentContext, PersonaDefinition, PlanState
from uclone_x.agent.prompt_assembler import (
    AnchorWriter,
    LiveAnchorProvenance,
    persisted_anchor_provenance,
    restored_anchor_provenance,
)
from uclone_x.agent.request_record import EpochRecordError, LogReader
from uclone_x.agent.session import (
    ContextSnapshot,
    SessionState,
    redact_message,
    validate_session_id,
)
from uclone_x.agent.tool_invoker import ToolInvoker
from uclone_x.core.context_state import (
    EPOCH_RESTORED,
    ContextEntry,
    ContextEpoch,
    advance,
    opening_entries,
    rendered_message,
)
from uclone_x.core.session_log import (
    LoggedMessage,
    SessionLogEntry,
    SessionLogKind,
    SessionLogProvenance,
    history_entry_ids,
    is_kept_text,
    is_result_message,
    kept_result_text,
    logged_message,
    logged_text,
    new_entry,
    stored_result_entry,
    tool_result_kind,
    with_image_data,
)
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import result_handle
from uclone_x.engine.event_bus import UnauthorizedSubscriptionError
from uclone_x.engine.protocols import EventSubscriptionProtocol
from uclone_x.errors import (
    FormTextMismatchError,
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


def _no_stored_body(_digest: str) -> str | None:
    """No store: a session held only in memory has no body outside `pending_bodies`."""
    return None


@dataclass(slots=True)
class _LiveSession:
    """Mutable working copy of one session's conversation.

    The history is not held: it is derived from the log and the last epoch (#1848). Its
    entries are what the last request showed -- the anchor, then the last epoch's entries
    -- followed by every message logged since (`entry_ids`). Only between a write that
    does not append -- a replace, a cut, a compaction, a rollback, a load -- and the next
    request is the list the write left held (`edited`), because no epoch records it yet.
    `messages` is each entry's logged body, decoded, with a form rendered from the kept
    result it records. So there is no second copy of the history to keep in sync with
    the log: what a request shows is what the log and the epoch say, by construction.
    `SessionState` is the frozen shape the store persists.

    **Every write to the history goes through one door, and logs as it writes.** `append`
    logs each message as a new entry, and that is all it does: a logged message is in the
    history. `replace` and `truncate` change or cut entries, and
    declare the cause they are given when the current epoch showed an entry they change
    (design §5.8, Rule 1: within an epoch the context only appends). `replace_history`
    puts a whole history in place -- a compaction, a rollback -- keeping the entry of each
    message the history already held. So the log cannot fall behind the history, and a
    writer cannot rewrite what a request showed without saying why.

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
    #: anchor instead of attributing every restored anchor to the caller. A record with no
    #: provenance on it comes back `UNRECORDED`, and is not re-resolved.
    anchor_provenance: LiveAnchorProvenance = AnchorWriter.CALLER
    #: What each turn's requests carried besides the conversation (#1421). Persisted as
    #: `SessionState.context_snapshots`; the last one is reused while nothing in it changes.
    context_snapshots: list[ContextSnapshot] = field(default_factory=list[ContextSnapshot])
    #: The conversation the previous request of this session sent, as the log entries and
    #: forms it showed (#2013), so the next `REQUEST_CONTEXT` records only what was added.
    #: Not persisted: after a restart the first request records its whole conversation
    #: once and the chain starts again.
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
    #: Every message that entered the history, append-only (#1443). Persisted as
    #: `SessionState.session_log`; each entry's body waits in `pending_bodies` like a
    #: snapshot layer's does. Written only by `_log` and `log_entry`.
    session_log: list[SessionLogEntry] = field(default_factory=list[SessionLogEntry])
    #: The history's entries as the last write that did not append left them (#1848):
    #: a replace, a cut, a compaction, a rollback or a load. `None` while the history is
    #: the last request's (`head` and the last epoch's entries); cleared by `record_shown`.
    #: Messages logged after `mark` follow it either way. Not persisted as such: the
    #: record names each message's entry (`SessionState.history_entries`).
    edited: list[str] | None = None
    #: The entries of the history the last request showed before its epoch's entries --
    #: the anchor, when the history holds one. Set by `record_shown`.
    head: tuple[str, ...] = ()
    #: How much of the log the history's base (`edited`, or `head` and the last epoch)
    #: accounts for; each message logged after it is in the history, after the base.
    mark: int = 0
    #: `messages` as last derived from the log; `None` after any write.
    view: tuple[ChatMessage, ...] | None = None
    #: What each request of this session showed, per epoch (#1443, design §5.8).
    #: Persisted as `SessionState.context_epochs`. Written only by `record_shown`.
    context_epochs: list[ContextEpoch] = field(default_factory=list[ContextEpoch])
    #: Why the next request may not extend the current epoch, as declared since the last
    #: request: a compaction, a rollback, a retry, a replaced or restored history. The
    #: next `record_shown` that opens an epoch records them, and every request clears them.
    #: Persisted as `SessionState.epoch_causes`, all but `restored`, so an epoch a
    #: compaction opens is still named for it after a restart (#1848).
    epoch_causes: list[str] = field(default_factory=list[str])
    #: Log bodies already decoded back into messages, by digest (`messages`). Not
    #: persisted: a cache of `ChatMessage.model_validate_json` over bodies the log holds.
    #: A form is held as it is logged, with no text (#1848).
    decoded: dict[str, ChatMessage] = field(default_factory=dict[str, ChatMessage])
    #: Forms already rendered for `messages`, by the digest of their logged body (#1848).
    #: A form's text is a function of that body and the kept result its handle names, so
    #: a rendering stays valid while the result does; a write re-derives `messages`
    #: without reading every kept result again. Not persisted, and cleared with the
    #: bodies (`forget_result_bodies`). A request does not read it: it renders through a
    #: `LogReader`, so a result lost since the last request is still refused (#1971).
    shown: dict[str, ChatMessage] = field(default_factory=dict[str, ChatMessage])
    #: Kept tool results already read for a request, by `tr_` handle (#1971). Not
    #: persisted: a cache over bodies this session's log holds, which every `LogReader`
    #: of the session shares. Cleared as each request's reader is made (`log_reader`), so
    #: each request reads a kept result from the store once and a body lost since the
    #: last request -- between turns or within one -- is refused, not served from memory
    #: (#1971, item 7); and when the bodies are forgotten.
    results: dict[str, str] = field(default_factory=dict[str, str])
    #: Per log entry of a compacted history, the entry it shows and its form, derived at
    #: the compaction (`compacted_entries`, #1848), for the request that opens the new
    #: epoch. A pruned message is the rendering of the entry it replaced. Set by the
    #: compaction driver, and by a rollback to the renderings of the checkpoint's epoch;
    #: cleared by `record_shown`. A request reads only these and the last epoch
    #: (`opening_entries`), never an earlier one. Persisted as
    #: `SessionState.compacted_entries`, keyed here by `ContextEntry.body`, so a restart
    #: before that request shows the same entries and renderings.
    compacted_entries: dict[str, ContextEntry] = field(default_factory=dict[str, ContextEntry])
    #: Reads a body this session wrote to the store, by digest (#1848): what a form is
    #: rendered from once its kept result has left `pending_bodies`. Set by `from_state`.
    load_body: Callable[[str], str | None] = field(default=_no_stored_body)

    @classmethod
    def from_state(
        cls,
        state: SessionState,
        *,
        anchor_provenance: LiveAnchorProvenance,
        load_body: Callable[[str], str | None],
    ) -> _LiveSession:
        """Adopt a persisted or seeded session as the live working copy.

        `anchor_provenance` is keyword-only and has no default here, so a call that omits
        it does not type-check: deciding who composed the anchor is part of putting a
        session into `_sessions`, and a default would let a new call inherit that answer
        by omission. The field's own default exists for the dataclass, not for callers.

        The record names the log entry of each message (`history_entries`), and the
        history is those entries: each is checked to be its message, by digest, and a
        record whose message is not its entry's body is refused. A state built from
        messages alone -- a seed, a reset -- has none, and its messages are matched to its
        log by digest (`history_entry_ids`); one the log does not account for gets an
        entry at the record's turn. A form is held as what it records, with no text
        (`SessionState` refuses one that holds text), and matched by that (#1848).

        `load_body` reads a body of this session from the store, by digest; a form's text
        is rendered from the kept result it records through it (`reader`), only when the
        history is shown. Nothing here reads a body, so adopting a record cannot fail on
        one.

        Raises:
            EpochRecordError: A message of the record is not the body of the log entry the
                record names for it; the error carries plain copy for the person.
        """
        live = cls(
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
            load_body=load_body,
        )
        live._edit(live._adopted(state))
        return live

    def _adopted(self, state: SessionState) -> list[str]:
        """The entries `state`'s messages are in this session's log, logging any it lacks.

        By the ids the record names (`history_entries`), each checked against its message
        by digest -- identity, not a match on text; or, for a state that names none, by
        digest (`history_entry_ids`). Each message is remembered decoded, so the history
        is readable without the store.
        """
        rendered = [logged_message(redact_message(message)) for message in state.messages]
        if state.history_entries:
            for entry_id, item in zip(state.history_entries, rendered, strict=True):
                if self._entry(entry_id).digest != item.digest:
                    raise EpochRecordError(
                        "Part of this conversation's record could not be read, so it "
                        "cannot continue.",
                        detail=f"the message recorded for log entry {entry_id} is not its body",
                        code="unreadable",
                    )
                self._remember(item)
            return list(state.history_entries)
        known = history_entry_ids(self.session_log, [item.digest for item in rendered])
        ids: list[str] = []
        for entry_id, item in zip(known, rendered, strict=True):
            if entry_id is None:
                ids.append(self._log(item))
            else:
                self._remember(item)
                ids.append(entry_id)
        return ids

    def entries_of(self, state: SessionState) -> list[str | None]:
        """The log entry each of `state`'s messages is, without logging anything.

        The ids the record names; for a state that names none, a match by digest over its
        own log, `None` for a message it does not account for.
        """
        if state.history_entries:
            return list(state.history_entries)
        return history_entry_ids(
            state.session_log,
            [logged_message(redact_message(message)).digest for message in state.messages],
        )

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        """The history as messages: each entry's logged body, decoded and rendered (#1848).

        Derived, never assigned: a write goes through `append`, `replace`, `truncate` or
        `replace_history`. A message is what its logged body decodes to -- redacted, as the
        log holds it -- and an `excerpt` or `stub` is rendered from the kept result it
        records, since the log holds no text for it. So the history and the log cannot
        disagree.

        Raises:
            EpochRecordError: A kept result a form is rendered from is missing or does not
                match its handle; the error carries plain copy for the person.
        """
        if self.view is None:
            reader = self.reader()
            self.view = tuple(self._shown(entry, reader) for entry in self.entry_ids())
        return self.view

    @property
    def recorded(self) -> tuple[ChatMessage, ...]:
        """The history as the log holds it: each entry's body, decoded, forms with no text.

        What a saved record's messages are (#1848): a form is its form and the kept result
        it records, never its text, so a record cannot hold text other than what was sent.
        Nothing is rendered, so no kept result is read and this cannot fail on a missing
        one -- a snapshot, a checkpoint or a save of a session whose result was lost still
        succeeds, and the loss is refused when the history is next shown.
        """
        reader = self.reader()
        return tuple(reader.message_of(entry) for entry in self.entry_ids())

    def _entry(self, entry_id: str) -> SessionLogEntry:
        """The log entry `entry_id` names; ids are log positions (`e<n>`)."""
        return self.session_log[int(entry_id[1:])]

    def _remember(self, item: LoggedMessage) -> None:
        """Hold `item`'s body decoded, so the history reads it without the store.

        With its image bytes (#2107): the body names each image by digest only, so they
        are put back from `item` or, for a message read back without them, from the
        bodies this session holds.
        """
        if item.digest not in self.decoded:
            message = ChatMessage.model_validate_json(item.body)
            held = dict(item.images)
            if held:
                message = message.model_copy(
                    update={
                        "images": tuple(
                            part.model_copy(update={"data": held[part.digest]})
                            if part.digest in held
                            else part
                            for part in message.images
                        )
                    }
                )
            self.decoded[item.digest] = with_image_data(message, self._load_any)

    def _load_any(self, digest: str) -> str | None:
        """The body `digest` names: not yet written (`pending_bodies`), else the store's."""
        body = self.pending_bodies.get(digest)
        return body if body is not None else self.load_body(digest)

    def _queue_images(self, item: LoggedMessage) -> None:
        """Queue `item`'s image bytes as bodies of their own, each named by its digest.

        Kept once per session however often the picture is logged, and never inside a
        message's body or the record, as a full tool result is kept (#1848, #2107).
        """
        for digest, data in item.images:
            if digest not in self.stored_bodies:
                self.pending_bodies[digest] = data

    def _shown(self, entry_id: str, reader: LogReader) -> ChatMessage:
        """The message `entry_id` is, with a form's text rendered from its kept result."""
        message = reader.message_of(entry_id)
        if message.rendered_from is None:
            return message
        digest = self._entry(entry_id).digest
        shown = self.shown.get(digest)
        if shown is None:
            shown = rendered_message(message, reader.result_of)
            self.shown[digest] = shown
        return shown

    def reader(self) -> LogReader:
        """What renders this session's requests from its log (#1848).

        The `LogReader` the log-only rebuild renders an epoch with, reading a body from
        `pending_bodies` first and then the store, and sharing this session's decoded
        messages and kept results read so far (#1971).
        """

        return LogReader(
            self.session_log,
            self._load_any,
            purpose="send",
            decoded=self.decoded,
            results=self.results,
        )

    def _logged(self, message: ChatMessage) -> LoggedMessage:
        """`message` as the log holds it, refusing a form whose text is not its rendering.

        A form is logged as what it was cut from, not as its text (`logged_message`), so a
        writer that hands in text the form does not render to -- a cap it did not record,
        say -- would send something other than what it wrote. That is a bug in the writer,
        and it fails here, loudly, rather than being sent as the other text (#1848).

        Raises:
            FormTextMismatchError: A form's text is other than its rendering, or the form
                records no result it was rendered from. Its message is plain; `detail`
                names the form and its handle, for the log (#1974).
        """
        clean = redact_message(message)
        if clean.form is not None and clean.rendered_from is None:
            # A form the writer cut from nothing kept -- a compactor with no session's
            # bodies truncates so (#1974) -- has no record to render it from, so the log
            # cannot keep it; refused in plain words, not the log's own error.
            raise FormTextMismatchError(
                "Part of this conversation could not be saved as it was written, so "
                "nothing was changed.",
                detail=f"a {clean.form} that records no result it was rendered from (#1974)",
            )
        item = logged_message(clean)
        if clean.rendered_from is not None and clean.content is not None:
            shown = rendered_message(clean, self.reader().result_of).content
            if shown != clean.content:
                raise FormTextMismatchError(
                    "Part of this conversation could not be saved as it was written, so "
                    "nothing was changed.",
                    detail=(
                        f"a {clean.form} written as text other than its rendering from "
                        f"{clean.rendered_from.handle} (#1848)"
                    ),
                )
        return item

    def as_recorded(self, message: ChatMessage) -> ChatMessage:
        """`message` as a record holds it: a form with no text, once its text is checked.

        For messages a caller hands in (`load_history`): a form that carries text is held
        as what it records, and its text must be its rendering (`_logged`), so a caller
        cannot put text into the history that the record would not show.

        Raises:
            FormTextMismatchError: A form's text is other than its rendering.
        """
        if message.form is None or message.rendered_from is None or message.content is None:
            return message
        self._logged(message)
        return message.model_copy(update={"content": None})

    def _log(self, item: LoggedMessage) -> str:
        """Log `item` as a new entry at the session's turn; the entry's id."""
        logged = new_entry(
            len(self.session_log),
            item,
            turn=self.turn_counter,
            provenance=SessionLogProvenance.RECORDED,
        )
        self.session_log.append(logged)
        if item.digest not in self.stored_bodies:
            self.pending_bodies[item.digest] = item.body
        self._queue_images(item)
        self._remember(item)
        return logged.id

    def _edit(self, entries: list[str]) -> None:
        """Hold `entries` as the history until the next request records it (`edited`)."""
        self.edited = entries
        self.mark = len(self.session_log)
        self.view = None

    def _shows_any(self, entries: Sequence[str]) -> bool:
        """Whether the current epoch shows any of `entries`.

        A message a compaction pruned is a new body that shows an entry of the epoch in a
        smaller form (`compacted_entries`); until the next request records it, it counts
        as shown when the entry it shows was (#1971, item 6).
        """
        if not self.context_epochs or not entries:
            return False
        epoch = self.context_epochs[-1].entries
        bodies = {shown.body for shown in epoch}
        ids = {shown.entry for shown in epoch}

        def shown(entry: str) -> bool:
            if entry in bodies:
                return True
            derived = self.compacted_entries.get(entry)
            return derived is not None and derived.entry in ids

        return any(shown(entry) for entry in entries)

    def append(self, *messages: ChatMessage) -> None:
        """Add `messages` to the end of the history, each logged as a new entry (#1443).

        A message that entered is logged whatever its text, so one identical to a message
        that left -- a prompt retried after a rollback -- is logged again, because it
        entered again. An append only extends the epoch, so it declares nothing (Rule 1).
        """
        for message in messages:
            self._log(self._logged(message))
        self.view = None

    def replace(self, index: int, message: ChatMessage, *, cause: str) -> None:
        """Put `message` in place of the history's message at `index` (#1848).

        The message is logged as a new entry; the one it replaces stays in the log. When
        the current epoch showed that entry, the next request cannot extend the epoch, so
        `cause` is declared for it. Rewriting what no request has shown -- the final
        answer the model just returned -- declares nothing: the next request only appends
        it (#1854). A message whose logged body equals the one there changes nothing.
        """
        item = self._logged(message)
        entries = self.entry_ids()
        entry = entries[index]
        if item.digest == self._entry(entry).digest:
            return
        if self._shows_any((entry,)):
            self.declare_new_epoch(cause)
        entries[index] = self._log(item)
        self._edit(entries)

    def truncate(self, length: int, *, cause: str) -> None:
        """Cut the history to its first `length` messages (#1848).

        What leaves stays in the log. When the current epoch showed any of it, `cause` is
        declared for the next request; a step no request carried -- one refused, or one
        whose calls nothing answered -- leaves without declaring anything.
        """
        entries = self.entry_ids()
        if length >= len(entries):
            return
        if self._shows_any(entries[length:]):
            self.declare_new_epoch(cause)
        self._edit(entries[:length])

    def replace_history(self, messages: Sequence[ChatMessage], *, cause: str) -> None:
        """Put `messages` in place of the whole history (#1848).

        For a compaction or a replaced history. Each message is matched to an entry of the
        history held now by digest, in order: a message that stayed keeps its entry, and
        one the history does not hold -- a compaction's summary, a message that left and
        came back -- is logged as a new entry. `cause` is declared unless the new history
        only extends the one it replaces, which is still one epoch (Rule 1).
        """
        before = self.entry_ids()
        pools: dict[str, list[str]] = {}
        for entry in reversed(before):
            pools.setdefault(self._entry(entry).digest, []).append(entry)
        entries: list[str] = []
        for message in messages:
            item = self._logged(message)
            pool = pools.get(item.digest)
            entries.append(pool.pop() if pool else self._log(item))
        self._edit(entries)
        if entries[: len(before)] != before:
            self.declare_new_epoch(cause)

    def restore_history(self, checkpoint: SessionState) -> None:
        """Put `checkpoint`'s history back, as the entries it was then, and the epoch it
        was shown from (#1848).

        A rollback's history is the checkpoint's entries, by id: the record names the
        entry of each message (`history_entries`), each checked to be its message by
        digest, so a stub a compaction in the undone turn summarized away keeps its entry
        and its rendering, and a message identical to one the undone turn added is not
        mistaken for it -- entries are told apart by identity, never by comparing text
        (#1974, item 12). The log only grows, so the checkpoint's log is a prefix of this
        one, entry for entry; when it is not, the history is matched as any replaced one
        is (`replace_history`). The opening state of the next epoch is the epoch in force
        at the checkpoint (`opening_entries`): a request reads only the last epoch, which
        may be one the undone turn recorded. Declares `rollback`, even when only what no
        request showed was cut, so the epoch record names every rewind.
        """
        log = checkpoint.session_log
        if list(log) == self.session_log[: len(log)]:
            self._edit(self._adopted(checkpoint))
        else:
            self.replace_history(checkpoint.messages, cause="rollback")
        self.declare_new_epoch("rollback")
        self.compacted_entries = opening_entries(
            checkpoint.context_epochs,
            {shown.body: shown for shown in checkpoint.compacted_entries},
        )

    def log_entry(self, rendered: LoggedMessage) -> SessionLogEntry:
        """Log something a request sent that is not a history message (#1849).

        The recalled memory section of a turn is one: it is sent in the `[Turn Context]`
        tail, never in the history, so no history write logs it. Each history message is
        logged as it enters, so the entry follows the message it was recalled for. Nothing
        a request sends is changed: the entry is only a record, and its body waits in
        `pending_bodies` like any other.
        """
        logged = new_entry(
            len(self.session_log),
            rendered,
            turn=self.turn_counter,
            provenance=SessionLogProvenance.RECORDED,
        )
        self.session_log.append(logged)
        if rendered.digest not in self.stored_bodies:
            self.pending_bodies[rendered.digest] = rendered.body
        self._queue_images(rendered)
        return logged

    def keep_result_body(self, text: str, *, kind: SessionLogKind) -> str:
        """Log `text` as an entry whose body is the whole redacted text; its handle (#1848).

        The body waits in `pending_bodies` like every other and is written to the
        session's context body store before the record that names it, so a full tool
        result lives where the rest of the session does and goes with it. The handle is
        the first 16 hex digits of the body's digest. Text this log already holds under
        that handle is not logged again.

        Nor is text the history already holds whole (#2013): when compaction stubs a
        tool result that fit, the entry of that message has the full text as its body,
        and the handle names it -- the first 16 hex digits of the message's digest -- so
        the result is stored once.
        """
        whole = self._whole_result(text, kind)
        if whole is not None:
            return "tr_" + whole.digest[:16]
        rendered = logged_text(kind, text, blob=None)
        handle = result_handle(rendered.body)
        known = stored_result_entry(self.session_log, handle)
        if known is None or known.digest != rendered.digest:
            self.log_entry(logged_text(kind, text, blob=handle))
        return handle

    def _whole_result(self, text: str, kind: SessionLogKind) -> SessionLogEntry | None:
        """The entry of a history message that is this tool result whole, if any (#2013)."""
        for entry_id, message in reversed(self.logged_history()):
            if message.form is not None or message.content != text:
                continue
            entry = self._entry(entry_id)
            if is_result_message(entry) and entry.kind is kind:
                return entry
        return None

    def forget_result_bodies(self) -> None:
        """Drop the full tool results not yet written, as a delete removes the written ones.

        After a delete a handle into the session is refused, whether or not its body had
        reached the store (#1848). The entries stay; with no body they resolve to nothing.
        """
        for entry in self.session_log:
            if is_kept_text(entry) or is_result_message(entry):
                self.pending_bodies.pop(entry.digest, None)
        self.results.clear()
        self.shown.clear()
        self.view = None  # nor served from the view rendered before the delete (#1974)

    def result_body(self, handle: str, load: Callable[[str], str | None]) -> str | None:
        """The full text `handle` names in this session's log, or `None` (#1848).

        Read from `pending_bodies` while it is not written yet, else through `load`, the
        store's reader of a body by digest. A body that no longer hashes to its handle is
        not returned: what `tool_result_read` gives back is what the handle names.
        """
        entry = stored_result_entry(self.session_log, handle)
        if entry is None:
            return None
        body = self.pending_bodies.get(entry.digest)
        if body is None:
            body = load(entry.digest)
        if body is None or result_handle(body) != handle:
            return None
        return kept_result_text(entry, body)

    def entry_ids(self) -> list[str]:
        """The log entry each message of the history is, derived from the log (#1848).

        The base is what the last write that did not append left (`edited`), or else what
        the last request showed: its `head` and the last epoch's entries, each the body it
        was shown as. Every message logged since (`mark`) follows; a kept text -- a full
        tool result, a recalled memory -- is logged as a record, not a message, so it is
        not in the history.
        """
        if self.edited is not None:
            entries = list(self.edited)
        else:
            entries = list(self.head)
            if self.context_epochs:
                entries.extend(shown.body for shown in self.context_epochs[-1].entries)
        entries.extend(
            entry.id for entry in self.session_log[self.mark :] if not is_kept_text(entry)
        )
        return entries

    def logged_history(self) -> list[tuple[str, ChatMessage]]:
        """Each message of the history, rendered as `messages` is, with its log entry (#1848)."""
        return list(zip(self.entry_ids(), self.messages, strict=True))

    def shown_in_epoch(self, index: int) -> bool:
        """Whether the message at `index` of `messages` is an entry the current epoch shows.

        A message no request of this epoch has shown -- a final answer the model just
        returned -- can still be rewritten without breaking Rule 1: the next request only
        appends it, so it does not declare a new epoch (#1854).
        """
        entries = self.entry_ids()
        if not 0 <= index < len(entries):
            return False
        return self._shows_any((entries[index],))

    def declare_new_epoch(self, cause: str) -> None:
        """Say that the next request may show the history differently, and why (Rule 1)."""
        if cause not in self.epoch_causes:
            self.epoch_causes.append(cause)

    def record_shown(self, shown: list[ContextEntry], *, step: int) -> ContextEpoch:
        """Record what a request's conversation showed; returns the epoch it belongs to.

        The epoch is extended when the request only appended to it, and a new one opens
        otherwise, naming the causes declared since the last request (`declare_new_epoch`).
        From here the history is derived from that epoch (`entry_ids`): what the request
        showed must be the history's last entries, or the request showed something other
        than the history, which fails here rather than being recorded (#1848).

        Raises:
            RuntimeError: `shown` is not the history's last entries -- a bug in whatever
                built the request, never a state a person can reach.
        """
        entries = self.entry_ids()
        start = len(entries) - len(shown)
        if start < 0 or entries[start:] != [each.body for each in shown]:
            raise RuntimeError(
                "a request showed entries other than the history's last ones (#1848)"
            )
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
        self.head = tuple(entries[:start])
        self.edited = None
        self.mark = len(self.session_log)
        self.view = None
        return self.context_epochs[-1]

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
            # Each message as the log holds it: a form records its source, not its text,
            # so the record is never rendered and cannot fail on a lost result (#1848).
            messages=self.recorded,
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
            # Which entry each message is, so a load adopts them by id (#1848).
            history_entries=tuple(self.entry_ids()),
        )


@dataclass(frozen=True, slots=True)
class SessionResultBodies:
    """One session's full tool-result bodies, as `ResultBodies` (#1848).

    Kept as entries of the live session's log (`_LiveSession.keep_result_body`) and read
    back from it, the pending bodies first and then the session's context body store.
    """

    live: _LiveSession
    load: Callable[[str], str | None]

    def keep(self, text: str, *, tool_name: str | None) -> str:
        """Keep `text` in full, logged as the kind `tool_name`'s results are; its handle."""
        return self.live.keep_result_body(text, kind=tool_result_kind(tool_name))

    def read(self, handle: str) -> str | None:
        """The body `handle` names in this session, or `None`."""
        return self.live.result_body(handle, self.load)


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

    def stored_body_loader(self, session_id: str) -> Callable[[str], str | None]:
        """A reader of `session_id`'s bodies in the store, by digest, or of none without one.

        The store is looked up on each read, so an agent whose store is swapped after the
        session was loaded reads the store it has now.
        """

        def load(digest: str) -> str | None:
            store = self._store
            return None if store is None else store.load_context_body(session_id, digest)

        return load

    def result_bodies(self, session_id: str) -> SessionResultBodies:
        """`session_id`'s full tool-result bodies, read from its log and its store (#1848)."""
        live = self._live_session(session_id)
        return SessionResultBodies(live, self.stored_body_loader(session_id))

    def log_reader(self, session_id: str) -> LogReader:
        """What renders `session_id`'s next request from its log (#1848).

        The same `LogReader` the log-only rebuild renders an epoch with, reading a body
        from the pending bodies first and then the store, and sharing the session's
        decoded messages. So the request sent is the rendering a rebuild produces.

        One per request, and each reads its kept results from the store afresh: a result
        kept from an earlier request is not sent from memory, so one removed since -- even
        within a turn -- is refused here as a rebuild would refuse it (#1971, item 7).
        Within the request each is read once.
        """
        live = self._live_session(session_id)
        live.results.clear()
        return live.reader()

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
            load_body=self.stored_body_loader(session_id),
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
            FormTextMismatchError: A form among `messages` carries text other than its
                rendering from the kept result it records (#1848); nothing is changed.
        """
        sid = self._effective_session_id(session_id)
        # "load history into", not "hydrate": the label is interpolated into the refusal,
        # and telling a caller that called `load_history` that it "cannot hydrate" names
        # something other than what it did. Same species as the `switch_session` message
        # that named a remedy which did not exist.
        self._refuse_session_mutation_during_turn(sid, "load history into")
        live = self._live_session(sid)
        # A form a caller hands in with its text is held as what it records (#1848).
        state = live.to_state(sid, self._config.agent_id).with_messages(
            [live.as_recorded(message) for message in messages],
            turn_counter=live.turn_counter if turn_counter is None else turn_counter,
        )
        # The caller composed these messages, so their anchor is not this agent's to
        # re-resolve on a later turn (#1081). See `_anchor_is_stale`.
        replaced = _LiveSession.from_state(
            state, anchor_provenance=AnchorWriter.CALLER, load_body=self.stored_body_loader(sid)
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
        # Compared by entry, not by text (#1974, item 12): the prefix stayed only if the
        # history still begins with the very entries the checkpoint's messages were.
        entries = live.entry_ids()
        prefix_rewritten = entries[:kept] != live.entries_of(checkpoint)
        dropped = len(entries) - kept if not prefix_rewritten else len(entries)
        if dropped or prefix_rewritten:
            # The undone turn's messages leave the history, not the log: each was logged
            # as it entered. A rollback always opens a new epoch, even one that only cut
            # what no request showed, so the epoch record names every rewind.
            live.restore_history(checkpoint)
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
        process rather than being undone by the next hydrate. That save sets an unreadable
        record aside first (#1844). While a copy of the record is kept -- set aside by this
        save or an earlier one -- the event log, context bodies and tool artifacts stay,
        because the copy names them too and restored by hand would lose its tool results
        without them (#1921). The record and memory are reset either way.

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
        load_body = self.stored_body_loader(sid)
        self._sessions[sid] = _LiveSession.from_state(
            reset, load_body=load_body, anchor_provenance=self._resolved_persona()
        )
        self._pending_durable_events = [
            event for event in self._pending_durable_events if event.get("session_id") != sid
        ]
        self._session_compactors.pop(sid, None)
        if sid == self._context.session_id:
            self._run_steps = 0
            self._loaded_skills.clear()  # reset active session skills
        if self._keeps_copy_of(sid):
            # The save above, or an earlier one, kept an unreadable record of this session
            # aside, and that copy names this session's event log, context bodies and tool
            # results. A reset clears the conversation this build could read, not the
            # copy's: they stay while it does, as `SessionStore.delete` keeps them (#1921).
            logger.info(
                "Session %r reset; its event log and tool results are kept for the copy of "
                "its earlier record set aside beside it",
                sid,
            )
            return reset
        if self._store is not None:
            # The event log holds the cleared conversation's tool outputs; a reset that
            # kept it would leave them on disk and continue the next conversation in the
            # same log. Last, after the record is reset, so a refused reset keeps it (#1442).
            # The full tool results are bodies of that store too (#1848), so a handle into
            # the cleared conversation does not resolve in the next one.
            self._store.clear_event_log(sid)
        return reset

    def _keeps_copy_of(self, sid: str) -> bool:
        """Whether the store keeps a set-aside copy of `sid`'s record (#1921).

        Only a store that sets records aside can say so; any other keeps none. A store
        that cannot tell answers `True` (`SessionStore.keeps_copy_of`), and so does a
        question that raises: keeping what a copy may need is the side that can be undone.
        """
        keeps = getattr(self._store, "keeps_copy_of", None)
        if keeps is None:
            return False
        try:
            return keeps(sid) is True
        except Exception:
            logger.exception("Could not tell whether session %r has a kept copy", sid)
            return True

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
        # Every history write logged as it wrote, so the entries `to_state` is about to
        # record already have their bodies queued here (#1443, #1848).
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
        # A record written through `to_state` accounts for every message in its log
        # already, so this adds nothing to it.
        hydrated = _LiveSession.from_state(
            loaded, anchor_provenance=restored, load_body=self.stored_body_loader(sid)
        )
        hydrated.declare_new_epoch(EPOCH_RESTORED)
        self._tool_invoker.reseed_bound_tools_from_history(hydrated)
        self._sessions[sid] = hydrated
        return loaded

    def delete_session(self, session_id: str | None = None) -> bool:
        """Delete a session's record, with its bodies and full tool results (#1848)."""
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

        live = self._sessions.get(sid)
        if live is not None:
            live.forget_result_bodies()
        if self._store is not None:
            return self._store.delete(sid)
        return False
