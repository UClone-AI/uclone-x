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
from collections.abc import Iterable, Mapping, Sequence
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


def rebuild_requests(
    store: SessionStoreProtocol, state: SessionState, events: Iterable[Mapping[str, Any]]
) -> list[RebuiltRequest]:
    """Every request the session log records, rebuilt exactly as it was assembled.

    `events` is the session's log in order, as `uclone_x.log.reader.read_session_log`
    reads it; the caller reads it, so this module stays below the log adapter.

    Each `REQUEST_CONTEXT` event extends the conversation of the one before it in the log
    by `kept_message_count` and `appended_messages`, and names the snapshot in `state` that
    holds the rest. An event whose `base_request` is not the request before it in the log
    means a request is missing from the log, and the rebuild stops there rather than
    guessing.

    Raises:
        RequestRecordError: A snapshot, a body or a request the chain depends on is
            missing from the record.
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

    rebuilt: list[RebuiltRequest] = []
    conversation: list[dict[str, Any]] = []
    previous_seq: int | None = None
    for event in events:
        if event.get("type") != "REQUEST_CONTEXT":
            continue
        base = event.get("base_request")
        if base is not None and base != previous_seq:
            raise RequestRecordError(
                "Part of this conversation's record is missing, so a request in it cannot "
                "be rebuilt.",
                detail=f"request {event.get('request')} extends {base}, last seen {previous_seq}",
            )
        kept = int(event["kept_message_count"])
        conversation = conversation[:kept] + list(event["appended_messages"])
        previous_seq = event.get("request")
        snapshot_id = str(event["snapshot"])
        snapshot = snapshots.get(snapshot_id)
        if snapshot is None:
            raise RequestRecordError(
                "Part of this conversation's record is missing, so a request in it cannot "
                "be rebuilt.",
                detail=f"no context snapshot {snapshot_id}",
            )
        rebuilt.append(_rebuild_one(event, snapshot, conversation, body, body_intact))
    return rebuilt


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
    )
