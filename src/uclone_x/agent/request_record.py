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
from typing import Any

from uclone_x.agent.session import ContextSnapshot, SessionState, content_digest
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ToolDefinition

__all__ = [
    "RebuiltRequest",
    "RequestLayers",
    "RequestRecordError",
    "assemble_request_messages",
    "compose_system_message",
    "messages_digest",
    "place_turn_context",
    "rebuild_requests",
    "serialize_tools",
]


class RequestRecordError(UCloneXError):
    """A recorded request could not be rebuilt, because part of its record is missing.

    The message is for the person reading the record. What is missing is named on the
    exception's `detail`.
    """

    def __init__(self, message: str, *, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


def compose_system_message(identity: str, slow_context: str) -> str:
    """The system message a request sends: the identity layer, then the slow context.

    With no slow context the identity goes out unchanged, byte for byte.
    """
    return f"{identity.strip()}\n\n{slow_context}".strip() if slow_context else identity


def place_turn_context(messages: list[ChatMessage], block: str) -> list[ChatMessage]:
    """Put the turn-context `block` at the tail of `messages`; see `BaseAgent`'s `_turn_context_block`.

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
    is history without its anchor: the anchor is where the identity was stored, and the
    identity field is what the request sends in its place.
    """

    identity: str
    slow_context: str
    system_message: bool
    conversation: tuple[ChatMessage, ...]
    turn_context: str


def assemble_request_messages(layers: RequestLayers) -> list[ChatMessage]:
    """The messages a request sends, from its layers. The only place they are assembled."""
    messages = list(layers.conversation)
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
    by `kept_message_count` and `appended_messages`, and names the snapshot in `state` that
    holds the rest. An event whose `base_request` is not the request before it in the log
    means a request is missing from the log: every request folded on top of the gap would
    be a guess, so none of them is rebuilt. A request that keeps nothing of the one before
    it (`kept_message_count` 0: after a restart, or the first request after a rollback)
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
                )
            bodies[digest] = text
            body_intact[digest] = content_digest(text) == digest
        return bodies[digest]

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
            rebuilt.append(_rebuild_selected(event, snapshots, conversation, body, body_intact))
        except RequestRecordError as err:
            if on_error is None:
                raise
            on_error(step, err)
    return rebuilt


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
        kept = int(event["kept_message_count"])
        appended = list(event["appended_messages"])
    except (KeyError, TypeError, ValueError):
        return RequestRecordError(
            "Part of this conversation's record could not be read, so a request in it "
            "cannot be rebuilt.",
            detail=f"request {event.get('request')} has no readable conversation delta",
        )
    return kept, appended


def _rebuild_selected(
    event: Mapping[str, Any],
    snapshots: Mapping[str, ContextSnapshot],
    conversation: list[dict[str, Any]],
    body: Any,
    body_intact: dict[str, bool],
) -> RebuiltRequest:
    if event.get("snapshot") is None:
        raise RequestRecordError(
            "Part of this conversation's record is missing, so a request in it cannot be rebuilt.",
            detail="request recorded before request capture (#1421)",
        )
    snapshot_id = str(event["snapshot"])
    snapshot = snapshots.get(snapshot_id)
    if snapshot is None:
        raise RequestRecordError(
            "Part of this conversation's record is missing, so a request in it cannot be rebuilt.",
            detail=f"no context snapshot {snapshot_id}",
        )
    try:
        return _rebuild_one(event, snapshot, conversation, body, body_intact)
    except (ValueError, TypeError, KeyError) as exc:
        # A body or a message that does not parse: pydantic's ValidationError and
        # json's JSONDecodeError are both ValueErrors. Named by kind only -- the
        # exception text can quote the record, which is the reader's to open, not ours.
        raise RequestRecordError(
            "Part of this conversation's record could not be read, so a request in it "
            "cannot be rebuilt.",
            detail=f"request {event.get('request')} could not be read ({type(exc).__name__})",
        ) from exc


def _rebuild_one(
    event: Mapping[str, Any],
    snapshot: ContextSnapshot,
    conversation: list[dict[str, Any]],
    body: Any,
    body_intact: dict[str, bool],
) -> RebuiltRequest:
    tools_text = body(snapshot.tools_digest)
    layers = RequestLayers(
        identity=body(snapshot.identity_digest),
        slow_context=body(snapshot.slow_context_digest),
        system_message=snapshot.system_message,
        conversation=tuple(ChatMessage.model_validate_json(json.dumps(m)) for m in conversation),
        turn_context=body(snapshot.turn_context_digest),
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
    )
