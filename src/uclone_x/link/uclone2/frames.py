"""The WebSocket frames of the Linked Runtime contract, protocol 1, as this runtime uses them.

The contract is uClone2's `docs/contracts/linked-runtime/frames.schema.json`, owned by both
repositories. Frames from the server are read with models that ignore fields they do not
know, because the contract adds optional fields without a protocol bump. Frames to the
server are built by the functions below and nowhere else, so one contract test can check
every shape this runtime sends.

Nothing here touches the network or the token.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "PROTOCOL",
    "Bye",
    "ErrorFrame",
    "FrameError",
    "Hello",
    "Limits",
    "Ping",
    "TaskCancel",
    "TaskOffer",
    "ack_frame",
    "bye_logout_frame",
    "error_frame",
    "fail_frame",
    "parse_frame",
    "pong_frame",
    "ready_frame",
]

PROTOCOL: Final = 1

Frame = dict[str, object]


class _Inbound(BaseModel):
    """A frame from the server: frozen, and tolerant of fields a newer server adds."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class Limits(_Inbound):
    """Server-side length limits, in code points after trimming. An absent limit is none."""

    guestbook_reply_max_chars: int = Field(ge=1)
    post_title_max_chars: int | None = Field(default=None, ge=1)
    post_body_max_chars: int | None = Field(default=None, ge=1)
    comment_max_chars: int | None = Field(default=None, ge=1)
    vote_reasoning_max_chars: int | None = Field(default=None, ge=1)


class Hello(_Inbound):
    type: Literal["hello"]
    protocol: Literal[1]
    bot_id: str = Field(min_length=1)
    server_time: datetime
    pending: int = Field(ge=0)
    limits: Limits
    clone_updated_at: datetime

    def length_limits(self) -> dict[str, int]:
        """The limits the server sent, by field name; a limit it left out is not a key."""
        return self.limits.model_dump(exclude_none=True)


class Ping(_Inbound):
    type: Literal["ping"]
    ts: str = Field(min_length=1)


class Bye(_Inbound):
    type: Literal["bye"]
    reason: str


class ErrorFrame(_Inbound):
    type: Literal["error"]
    code: str
    message: str
    delivery_id: str | None = None


class TaskOffer(_Inbound):
    """Only what this step reads of an offer; the context is step 3's."""

    type: Literal["task.offer"]
    delivery_id: str = Field(min_length=1)
    task_type: str


class TaskCancel(_Inbound):
    type: Literal["task.cancel"]
    delivery_id: str = Field(min_length=1)
    reason: str


_INBOUND: Final[dict[str, type[_Inbound]]] = {
    "hello": Hello,
    "ping": Ping,
    "bye": Bye,
    "error": ErrorFrame,
    "task.offer": TaskOffer,
    "task.cancel": TaskCancel,
}

#: Frame types the contract gives the runtime to send. The server sending one is not an
#: unknown type, it is a frame out of place.
_OUTBOUND_TYPES: Final = frozenset({"ready", "pong", "task.ack", "task.complete", "task.fail"})


class FrameError(Exception):
    """A frame from the server could not be used. `code` is the contract's `error.code`."""

    def __init__(self, code: Literal["unknown_type", "invalid_frame"], detail: str) -> None:
        super().__init__(detail)
        self.code: Literal["unknown_type", "invalid_frame"] = code
        #: The frame's `type` when it had a readable one, for the Diagnostics line.
        self.detail = detail


def parse_frame(raw: str | bytes) -> _Inbound:
    """Read one text frame from the server, or raise `FrameError`.

    The detail never quotes the frame: an offer's context is other people's text.
    """
    if isinstance(raw, bytes):
        raise FrameError("invalid_frame", "binary frame")
    try:
        data = cast(object, json.loads(raw))
    except ValueError:
        raise FrameError("invalid_frame", "not JSON") from None
    if not isinstance(data, dict):
        raise FrameError("invalid_frame", "not an object")
    frame_type = cast(dict[str, object], data).get("type")
    if not isinstance(frame_type, str):
        raise FrameError("invalid_frame", "no type")
    model = _INBOUND.get(frame_type)
    if model is None:
        if frame_type in _OUTBOUND_TYPES:
            raise FrameError("invalid_frame", f"{frame_type} is a runtime frame")
        raise FrameError("unknown_type", "unknown frame type")
    try:
        return model.model_validate(data)
    except ValidationError:
        raise FrameError("invalid_frame", f"{frame_type} does not match the contract") from None


# --- frames this runtime sends --------------------------------------------------------


def ready_frame(*, max_concurrency: int, task_types: list[str]) -> Frame:
    """`task_timeout_s` is left out: the server's default (600 s) applies."""
    return {
        "type": "ready",
        "protocol": PROTOCOL,
        "max_concurrency": max_concurrency,
        "task_types": list(task_types),
    }


def pong_frame(ts: str) -> Frame:
    """Echoes the ping's `ts`, so the server can measure the round trip."""
    return {"type": "pong", "ts": ts}


def bye_logout_frame() -> Frame:
    """The runtime is ending the session on purpose: uClone2 shows the clone offline at once."""
    return {"type": "bye", "reason": "logout"}


def ack_frame(delivery_id: str) -> Frame:
    return {"type": "task.ack", "delivery_id": delivery_id}


FailCode = Literal[
    "target_gone",
    "write_rejected",
    "no_decision",
    "model_error",
    "runtime_error",
    "rate_limited",
    "server_error",
]

_REASON_MAX: Final = 500


def fail_frame(delivery_id: str, *, code: FailCode, retryable: bool, reason: str) -> Frame:
    """`reason` is shown to the clone's owner in uClone2: a plain sentence, ≤ 500 chars."""
    return {
        "type": "task.fail",
        "delivery_id": delivery_id,
        "retryable": retryable,
        "code": code,
        "reason": reason[:_REASON_MAX],
    }


def error_frame(
    code: Literal["unknown_type", "invalid_frame"], message: str, delivery_id: str | None = None
) -> Frame:
    frame: Frame = {"type": "error", "code": code, "message": message}
    if delivery_id:
        frame["delivery_id"] = delivery_id
    return frame
