"""How a request is assembled from its layers, and how a recorded one is rebuilt (#1421).

A request the agent sends is five layers: the tools, the identity prompt, the slow context
(invariants, skills, workspace), the conversation, and the turn context at the tail. The
agent builds a `RequestLayers` for each request and turns it into messages through
`assemble_request_messages`. Nothing else assembles a request.

The record keeps the same layers apart. A `ContextSnapshot` in the session holds the
hashes of the tools, identity, slow context and turn context, with their bodies stored once
in the session store, plus the model settings. Each `REQUEST_CONTEXT` event
in the session log names its snapshot and carries only the conversation messages added
since the previous request. `rebuild_requests` reads the three back and assembles each
request through the same `assemble_request_messages`, so a rebuilt request and a sent one
cannot be put together differently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from uclone_x.agent.k_act import text_call_messages
from uclone_x.agent.session import ContextSnapshot, SessionState, content_digest
from uclone_x.agent.tools_module import (
    DEFAULT_TOOLS_MODULE,
    ToolsModuleName,
    UnknownToolsModuleError,
    recorded_tools_module,
)
from uclone_x.core.context_state import (
    ContextEntry,
    ContextEpoch,
    render_entries,
    rendered_message,
)
from uclone_x.core.session_log import (
    SessionLogEntry,
    kept_result_text,
    stored_result_entry,
    with_image_data,
)
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import result_handle
from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ToolDefinition

__all__ = [
    "RebuiltRequest",
    "RequestLayers",
    "EpochErrorCode",
    "EpochRecordError",
    "LogReader",
    "LogRenderPurpose",
    "RecordErrorCode",
    "RequestRecordError",
    "assemble_request_messages",
    "compose_system_message",
    "epoch_renderer",
    "messages_digest",
    "place_turn_context",
    "rebuild_epoch_conversations",
    "rebuild_requests",
    "serialize_tools",
]


#: What kind of gap stopped a rebuild, as a stable code a head words in its reader's
#: language (#1907). A closed set: it names the kind only, never a path, a digest or an
#: exception's text -- `RequestRecordError.detail` carries the specifics, in English.
RecordErrorCode = Literal[
    "body_missing",
    "snapshot_missing",
    "log_entry_missing",
    "chain_broken",
    "before_capture",
    "unreadable",
    "epoch_mismatch",
]

#: The `RecordErrorCode`s rendering one epoch from the session log can raise (#1915): an
#: entry the log lacks, an entry's body missing or unparseable, or a form that is not its
#: entry's. The rest name a request's own record, which an epoch never reads.
EpochErrorCode = Literal["log_entry_missing", "body_missing", "unreadable", "epoch_mismatch"]


class RequestRecordError(UCloneXError):
    """A recorded request could not be rebuilt, because part of its record is missing.

    The message is for the person reading the record. What is missing is named on the
    exception's `detail`, and what kind of gap it is on its `code`.
    """

    def __init__(self, message: str, *, detail: str, code: RecordErrorCode) -> None:
        super().__init__(message)
        self.detail = detail
        self.code: RecordErrorCode = code


class EpochRecordError(RequestRecordError):
    """An epoch's conversation could not be rendered from the session log (#1915).

    `epoch_code` is the same code as `code`, typed to the four an epoch can raise, so a
    caller reporting it promises no more than an epoch can produce.
    """

    def __init__(self, message: str, *, detail: str, code: EpochErrorCode) -> None:
        super().__init__(message, detail=detail, code=code)
        self.epoch_code: EpochErrorCode = code


def compose_system_message(identity: str, slow_context: str) -> str:
    """The system message a request sends: the identity layer, then the slow context.

    With no slow context the identity goes out unchanged, byte for byte.
    """
    return f"{identity.strip()}\n\n{slow_context}".strip() if slow_context else identity


def place_turn_context(messages: list[ChatMessage], block: str) -> list[ChatMessage]:
    """Put the turn-context `block` at the tail of `messages`; see `agent.prompt_assembler.turn_context_block`.

    It joins the last message when that is the user's, so no two user messages are
    adjacent, and follows as a message of its own otherwise.
    """
    if not block:
        return messages
    last = messages[-1] if messages else None
    if last is not None and last.role is MessageRole.USER and last.content is not None:
        messages[-1] = last.model_copy(update={"content": f"{last.content}\n\n{block}"})
    else:
        messages.append(ChatMessage(role=MessageRole.USER, content=block))
    return messages


@dataclass(frozen=True)
class RequestLayers:
    """The layers of one request's messages, before they are put together.

    `identity` is the prompt as sent, already framed for the model family. `conversation`
    is history without its anchor, as the request renders it (a repeated tool result sent
    as a back-reference, `core/context_state.render_conversation`): the anchor is where
    the identity was stored, and the identity field is what the request sends in its
    place. `shown` is the session log entry and form of each conversation message, for
    the context state (#1443); a rebuilt request has none.
    """

    identity: str
    slow_context: str
    system_message: bool
    conversation: tuple[ChatMessage, ...]
    turn_context: str
    shown: tuple[ContextEntry, ...] = ()
    #: Whether the request sends calls as text (the ``k_act`` tools module, #2188): each
    #: assistant message then goes out as its text alone (`text_call_messages`).
    text_calls: bool = False


def assemble_request_messages(layers: RequestLayers) -> list[ChatMessage]:
    """The messages a request sends, from its layers. The only place they are assembled."""
    messages = list(layers.conversation)
    if layers.text_calls:
        messages = text_call_messages(messages)
    if layers.system_message:
        system = compose_system_message(layers.identity, layers.slow_context)
        messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=system))
    return place_turn_context(messages, layers.turn_context)


def serialize_tools(tools: Sequence[ToolDefinition]) -> str:
    """The tool layer as one byte-stable string: the schemas in the order sent.

    Keys sorted, no whitespace, non-ASCII kept as is, so the same list always hashes the
    same. The order is not sorted: it is part of what the model was shown.
    """
    return json.dumps(
        [tool.model_dump(mode="json") for tool in tools],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def messages_digest(messages: Sequence[dict[str, Any]]) -> str:
    """SHA-256 over a request's messages, in the canonical form `REQUEST_CONTEXT` records."""
    canonical = json.dumps(list(messages), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RebuiltRequest:
    """One request rebuilt from the record.

    `verified` says the rebuilt messages hash to the digest recorded when the request was
    sent, and every body hashes to its name. It is `False` when redaction changed a
    credential-shaped string on its way to disk: the request is then the record's
    redacted copy, not proven to be what was sent.
    """

    step: int
    snapshot_id: str
    request: LLMRequest
    verified: bool
    layers: RequestLayers | None = None
    #: The tools module the request was built under, as its snapshot records it (#2188).
    tools_module: ToolsModuleName = DEFAULT_TOOLS_MODULE


def rebuild_requests(
    store: SessionStoreProtocol,
    state: SessionState,
    events: Iterable[Mapping[str, Any]],
    *,
    select: Callable[[Mapping[str, Any]], bool] | None = None,
    on_error: Callable[[int, RequestRecordError], None] | None = None,
) -> list[RebuiltRequest]:
    """Every request the session log records, rebuilt exactly as it was assembled.

    `events` is the session's log in order, as `uclone_x.log.reader.read_session_log`
    reads it; the caller reads it, so this module stays below the log adapter.

    Each `REQUEST_CONTEXT` event extends the conversation of the one before it in the log
    by `kept_entry_count` and `appended_entries`, and names the snapshot in `state` that
    holds the rest. The conversation is recorded as the log entries it showed and their
    forms, and is rendered from `state.session_log` by the `LogReader` the live request
    renders through, so the event holds no message text (#2013). An event from before
    that, which records the messages themselves, is refused as unreadable, not rebuilt
    from its copy. An event whose `base_request` is not the request before it in the log
    means a request is missing from the log: every request folded on top of the gap would
    be a guess, so none of them is rebuilt. A request that keeps nothing of the one before
    it (`kept_entry_count` 0: after a restart, or the first request after a rollback)
    carries its whole conversation, so it starts the chain again and the requests from
    there on rebuild.

    When `select` is given, conversation is folded for every event up to the last
    selected one, but requests are only assembled and validated for events where
    `select(event)` is `True`. Nothing after the last selected event is read, so a gap
    later in the log does not touch the selected requests (#1490).

    When `on_error` is given, a request that cannot be rebuilt -- a gap before it, a
    missing snapshot or body, a pre-#1421 record, a record that does not parse -- is
    reported to it as `(step, RequestRecordError)` and the rebuild goes on with the next
    one. Without it the first such request raises.

    Raises:
        RequestRecordError: Without `on_error`: a snapshot, a body or a request the chain
            depends on is missing from the record, or a record does not parse.
    """
    snapshots = {snapshot.snapshot_id: snapshot for snapshot in state.context_snapshots}
    bodies: dict[str, str] = {}
    body_intact: dict[str, bool] = {}

    def body(digest: str) -> str:
        if digest not in bodies:
            text = store.load_context_body(state.session_id, digest)
            if text is None:
                raise RequestRecordError(
                    "Part of this conversation's record is missing, so a request in it "
                    "cannot be rebuilt.",
                    detail=f"no context body {digest}",
                    code="body_missing",  # a layer body a request's snapshot names
                )
            bodies[digest] = text
            body_intact[digest] = content_digest(text) == digest
        return bodies[digest]

    reader = LogReader(
        state.session_log,
        lambda digest: store.load_context_body(state.session_id, digest),
        purpose="rebuild",
    )
    requests = [event for event in events if event.get("type") == "REQUEST_CONTEXT"]
    if select is not None:
        chosen = [i for i, event in enumerate(requests) if select(event)]
        if not chosen:
            return []
        requests = requests[: chosen[-1] + 1]
        chosen_ids = {id(requests[i]) for i in chosen}
    else:
        chosen_ids = {id(event) for event in requests}

    rebuilt: list[RebuiltRequest] = []
    conversation: list[dict[str, Any]] = []
    previous_seq: int | None = None
    # Set once the chain is broken, and cleared by a request that restates everything.
    broken: RequestRecordError | None = None
    for event in requests:
        base = event.get("base_request")
        delta = _conversation_delta(event)
        if isinstance(delta, RequestRecordError):
            broken = delta
        elif delta[0] == 0:
            # Keeps nothing from before: its conversation is all in this event, so a gap
            # behind it cannot reach it (a restart, or the first turn after a rollback).
            broken = None
        elif base is not None and base != previous_seq:
            broken = RequestRecordError(
                "Part of this conversation's record is missing, so a request in it cannot "
                "be rebuilt.",
                detail=f"request {event.get('request')} extends {base}, last seen {previous_seq}",
                code="chain_broken",
            )
        if not isinstance(delta, RequestRecordError):
            kept, appended = delta
            conversation = conversation[:kept] + appended
        previous_seq = event.get("request")

        if id(event) not in chosen_ids:
            continue

        step = _step_of(event)
        try:
            if broken is not None:
                raise broken
            rebuilt.append(
                _rebuild_selected(event, snapshots, conversation, body, body_intact, reader)
            )
        except RequestRecordError as err:
            if on_error is None:
                raise
            on_error(step, err)
    return rebuilt


#: The two ends a log rendering serves, and how a gap in it is put to the person (#1848).
#: `rebuild` renders an earlier epoch for the record; `send` renders the request about to
#: go out, so a gap there stops the conversation rather than a view of it.
LogRenderPurpose = Literal["rebuild", "send"]

_MISSING: Final[dict[LogRenderPurpose, str]] = {
    "rebuild": "Part of this conversation's record is missing, so it cannot be rebuilt.",
    "send": "Part of this conversation's record is missing, so it cannot continue.",
}
_UNREADABLE: Final[dict[LogRenderPurpose, str]] = {
    "rebuild": "Part of this conversation's record could not be read, so it cannot be rebuilt.",
    "send": "Part of this conversation's record could not be read, so it cannot continue.",
}


class LogReader:
    """Renders a conversation from one session's log: the one way it is built (#1848).

    The request a turn sends (`PromptAssembler.prepare_turn_layers`) and the log-only
    rebuild of an epoch (`epoch_renderer`) both render through `render`, over the same
    entries and forms. They differ only in where a body is read: `load_body` gives the
    body a digest names -- the live session's pending bodies, then the store, for a
    request; the store alone for a rebuild. Bodies are content-addressed, so the same
    digest is the same text on both sides, and what is sent is what the rebuild renders.

    An entry is its body decoded (`message_of`). An `excerpt` or `stub` is logged as its
    form and what it was cut from, with no text, and is rendered from that result's full
    body (`result_of`, the log entry its handle names) with the parameters it records
    (`render_form`, `shown_message`). `decoded` and `results` may be shared across readers of one
    session: a body decodes to one message, and a handle names one text, whichever reader
    reads it -- so a kept result is read from the store once per session, not once per
    request (#1971). A shared `results` is cleared by whoever forgets the bodies.
    """

    def __init__(
        self,
        session_log: Sequence[SessionLogEntry],
        load_body: Callable[[str], str | None],
        *,
        purpose: LogRenderPurpose,
        decoded: dict[str, ChatMessage] | None = None,
        results: dict[str, str] | None = None,
    ) -> None:
        self._log = session_log
        self._load = load_body
        self._purpose: LogRenderPurpose = purpose
        self._decoded: dict[str, ChatMessage] = {} if decoded is None else decoded
        self._results: dict[str, str] = {} if results is None else results

    def _error(self, *, missing: bool, detail: str, code: EpochErrorCode) -> EpochRecordError:
        copy = _MISSING if missing else _UNREADABLE
        return EpochRecordError(copy[self._purpose], detail=detail, code=code)

    def _body(self, digest: str) -> str | None:
        """The body `digest` names, `None` when there is none; refused when it will not read.

        A body file that is there and cannot be read -- permissions, bad bytes -- is the
        record's gap like any other, so it is refused in the same plain words and never
        reaches a person as the operating system's text (#1974).
        """
        try:
            return self._load(digest)
        except (OSError, UnicodeDecodeError) as exc:
            raise self._error(
                missing=False,
                detail=f"context body {digest} could not be read ({type(exc).__name__})",
                code="unreadable",  # a body file that will not read
            ) from exc

    def result_of(self, handle: str) -> str:
        """The full body the kept result `handle` names."""
        if handle in self._results:
            return self._results[handle]
        entry = stored_result_entry(self._log, handle)
        if entry is None:
            raise self._error(
                missing=True,
                detail=f"no log entry for the kept result {handle}",
                code="log_entry_missing",
            )
        text = self._body(entry.digest)
        if text is None:
            raise self._error(
                missing=True,
                detail=f"no context body {entry.digest}",
                code="body_missing",  # a kept result's body
            )
        if result_handle(text) != handle:
            raise self._error(
                missing=False,
                detail=f"log entry {entry.id} is not the result {handle}",
                code="unreadable",  # a kept result's body does not hash to its handle
            )
        whole = kept_result_text(entry, text)
        if whole is None:
            raise self._error(
                missing=False,
                detail=f"log entry {entry.id} is not a whole tool result",
                code="unreadable",  # a message entry a handle names holds no full result
            )
        self._results[handle] = whole
        return whole

    def message_of(self, entry_id: str) -> ChatMessage:
        """The message log entry `entry_id` is: its body, decoded."""
        position = int(entry_id[1:]) if entry_id[:1] == "e" and entry_id[1:].isdigit() else -1
        if not 0 <= position < len(self._log):
            raise self._error(
                missing=True,
                detail=f"no log entry {entry_id}",
                code="log_entry_missing",  # an entry the epoch names
            )
        digest = self._log[position].digest
        if digest not in self._decoded:
            text = self._body(digest)
            if text is None:
                raise self._error(
                    missing=True,
                    detail=f"no context body {digest}",
                    code="body_missing",  # a log entry's body
                )
            try:
                message = ChatMessage.model_validate_json(text)
            except ValueError as exc:
                raise self._error(
                    missing=False,
                    detail=f"log entry {entry_id} could not be read ({type(exc).__name__})",
                    code="unreadable",  # a log entry's body does not parse
                ) from exc
            if message.form is not None and (
                message.rendered_from is None or message.content is not None
            ):
                # A form is logged as what it was cut from and no text (`logged_message`);
                # one that holds text, or records nothing to render from, was written in
                # an older format, and is not shown as whatever text it carries (#1848).
                raise self._error(
                    missing=False,
                    detail=f"log entry {entry_id} is a {message.form} not recorded as a form",
                    code="unreadable",  # a form logged as its text
                )
            # The body names each image by digest; its bytes are a body of their own
            # (#2107). One that is gone is sent as unavailable, never refused.
            self._decoded[digest] = with_image_data(message, self._load)
        return self._decoded[digest]

    def shown_message(self, entry_id: str) -> ChatMessage:
        """The message log entry `entry_id` is, with a form's text rendered (#1848)."""
        return rendered_message(self.message_of(entry_id), self.result_of)

    def render(self, entries: Sequence[ContextEntry], *, what: str) -> list[ChatMessage]:
        """The conversation `entries` render to; `what` names them in an error's detail."""
        try:
            return render_entries(entries, self.message_of, self.result_of)
        except ValueError as exc:
            # `render_entries` refuses a message whose form is not its entry's.
            raise self._error(
                missing=False,
                detail=f"{what} does not match its log entries",
                code="epoch_mismatch",
            ) from exc


def epoch_renderer(
    store: SessionStoreProtocol, state: SessionState
) -> Callable[[ContextEpoch], list[ChatMessage]]:
    """A function that renders one epoch's conversation from the session log alone (#1848).

    An epoch lists the entries its last request showed and their forms (§5.8); each is
    rendered by a `LogReader` over the record's log and the context-body store. No
    `REQUEST_CONTEXT` event and no `messages` is read, so this is the conversation the log
    and the context state say was sent -- the live request renders through the same
    `LogReader.render`. Bodies are decoded once across the epochs one renderer renders,
    and each epoch fails on its own: an unreadable body stops only the epochs that list
    it. No workspace is read.

    The returned function raises `EpochRecordError` when the epoch names an entry the
    log does not have, an entry's body is missing or does not parse, a kept result's
    entry is missing or its body does not hash to its handle, or a message's form is not
    the one its entry records.
    """
    reader = LogReader(
        state.session_log,
        lambda digest: store.load_context_body(state.session_id, digest),
        purpose="rebuild",
    )

    def render(epoch: ContextEpoch) -> list[ChatMessage]:
        return reader.render(epoch.entries, what=f"epoch {epoch.number}")

    return render


def rebuild_epoch_conversations(
    store: SessionStoreProtocol, state: SessionState
) -> list[list[ChatMessage]]:
    """Each epoch's conversation, rendered from the session log alone (#1848).

    See `epoch_renderer`, which renders one epoch.

    Raises:
        EpochRecordError: An epoch names an entry or a rendering the log does not have,
            an entry's body is missing or does not parse, a kept result a form is
            rendered from is missing or does not match its handle, or a message's form
            is not the one its entry records.
    """
    render = epoch_renderer(store, state)
    return [render(epoch) for epoch in state.context_epochs]


def _step_of(event: Mapping[str, Any]) -> int:
    try:
        return int(event.get("step", 0))
    except (TypeError, ValueError):
        return 0


def _conversation_delta(
    event: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]] | RequestRecordError:
    """What `event` keeps of the conversation before it and what it adds, or why not."""
    try:
        # A pre-#2013 event records `appended_messages`, the text itself, and has no
        # entries: it is refused here rather than rebuilt from that copy.
        kept = int(event["kept_entry_count"])
        appended = list(event["appended_entries"])
    except (KeyError, TypeError, ValueError):
        return RequestRecordError(
            "Part of this conversation's record could not be read, so a request in it "
            "cannot be rebuilt.",
            detail=f"request {event.get('request')} has no readable conversation delta",
            code="unreadable",  # a request's conversation delta
        )
    return kept, appended


def _rebuild_selected(
    event: Mapping[str, Any],
    snapshots: Mapping[str, ContextSnapshot],
    conversation: list[dict[str, Any]],
    body: Any,
    body_intact: dict[str, bool],
    reader: LogReader,
) -> RebuiltRequest:
    if event.get("snapshot") is None:
        raise RequestRecordError(
            "Part of this conversation's record is missing, so a request in it cannot be rebuilt.",
            detail="request recorded before request capture (#1421)",
            code="before_capture",
        )
    snapshot_id = str(event["snapshot"])
    snapshot = snapshots.get(snapshot_id)
    if snapshot is None:
        raise RequestRecordError(
            "Part of this conversation's record is missing, so a request in it cannot be rebuilt.",
            detail=f"no context snapshot {snapshot_id}",
            code="snapshot_missing",
        )
    try:
        return _rebuild_one(event, snapshot, conversation, body, body_intact, reader)
    except (ValueError, TypeError, KeyError) as exc:
        # A body or a message that does not parse: pydantic's ValidationError and
        # json's JSONDecodeError are both ValueErrors. Named by kind only -- the
        # exception text can quote the record, which is the reader's to open, not ours.
        raise RequestRecordError(
            "Part of this conversation's record could not be read, so a request in it "
            "cannot be rebuilt.",
            detail=f"request {event.get('request')} could not be read ({type(exc).__name__})",
            code="unreadable",  # a request's bodies or messages do not parse
        ) from exc


def _rebuild_one(
    event: Mapping[str, Any],
    snapshot: ContextSnapshot,
    conversation: list[dict[str, Any]],
    body: Any,
    body_intact: dict[str, bool],
    reader: LogReader,
) -> RebuiltRequest:
    try:
        tools_module = recorded_tools_module(snapshot.tools_module)
    except UnknownToolsModuleError as exc:
        # No migration (#2188): a module this build does not know is not read as another.
        raise RequestRecordError(
            "Part of this conversation's record could not be read, so a request in it "
            "cannot be rebuilt.",
            detail=f"snapshot {snapshot.snapshot_id} names an unknown tools module",
            code="unreadable",  # a snapshot's tools module this build does not have
        ) from exc
    tools_text = body(snapshot.tools_digest)
    shown = tuple(ContextEntry.model_validate_json(json.dumps(e)) for e in conversation)
    layers = RequestLayers(
        identity=body(snapshot.identity_digest),
        slow_context=body(snapshot.slow_context_digest),
        system_message=snapshot.system_message,
        conversation=tuple(reader.render(shown, what=f"request {event.get('request')}")),
        turn_context=body(snapshot.turn_context_digest),
        shown=shown,
        text_calls=tools_module == "k_act",
    )
    messages = assemble_request_messages(layers)
    request = LLMRequest(
        model=snapshot.model,
        messages=tuple(messages),
        tools=tuple(
            ToolDefinition.model_validate_json(json.dumps(t)) for t in json.loads(tools_text)
        ),
        temperature=snapshot.temperature,
        max_tokens=snapshot.max_tokens,
        auto_compact=snapshot.auto_compact,
        compaction_threshold_tokens=snapshot.compaction_threshold_tokens,
    )
    verified = (
        all(
            body_intact[d]
            for d in (
                snapshot.tools_digest,
                snapshot.identity_digest,
                snapshot.slow_context_digest,
                snapshot.turn_context_digest,
            )
        )
        and messages_digest([m.model_dump() for m in messages]) == event.get("digest")
        and len(messages) == event.get("message_count")
    )
    return RebuiltRequest(
        step=int(event["step"]),
        snapshot_id=snapshot.snapshot_id,
        request=request,
        verified=verified,
        layers=layers,
        tools_module=tools_module,
    )
