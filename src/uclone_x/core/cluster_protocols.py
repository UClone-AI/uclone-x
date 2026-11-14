"""Distributed runtime wire envelopes, error types, and adapter protocols (Kernel Layer)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.errors import UCloneXError

if TYPE_CHECKING:
    from datetime import datetime

    from uclone_x.core.session_state import SessionState
    from uclone_x.core.usage_entry import UsageEntry
    from uclone_x.engine.event_bus import AgentEvent
    from uclone_x.engine.protocols import EventBusProtocol
    from uclone_x.room.models import RoomState

__all__ = [
    "LANDED_ID_PREFIX",
    "LEARN_JOB_ID_PREFIX",
    "SEAT_TURN_ID_PREFIX",
    "VERSIONED_HEAD_SET_SCRIPT",
    "CacheClientProtocol",
    "ClusterRoomStoreProtocol",
    "ClusterSessionStoreProtocol",
    "CommandBrokerProtocol",
    "DeadLetterEnvelope",
    "DistributedUsageStoreProtocol",
    "InterruptControlEnvelope",
    "LearnCommandEnvelope",
    "RoomCommandEnvelope",
    "RoomStepOutbox",
    "RoomStoreProtocol",
    "SSEStreamEnvelope",
    "SSEStreamPublisherProtocol",
    "SqlConnProtocol",
    "SeatTurnRef",
    "SqlPoolProtocol",
    "TokenBudgetSnapshot",
    "TurnCommandEnvelope",
    "TurnLeaseConflictError",
    "TurnLeaseProtocol",
    "TurnLeaseToken",
    "TurnPreemptedError",
    "decode_versioned_head",
    "encode_versioned_head",
    "is_worker_id",
    "wire_id_problem",
    "wire_text_problem",
]


class TurnLeaseConflictError(UCloneXError):
    """Raised when another active worker already holds the execution lease for a conversation."""

    def __init__(self, conversation_id: str, turn_id: str, holder: str | None = None) -> None:
        detail = f" (holder={holder})" if holder else ""
        super().__init__(
            f"Turn lease conflict for conversation '{conversation_id}' on turn '{turn_id}'{detail}"
        )
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.holder = holder


class TurnPreemptedError(UCloneXError):
    """Raised when a turn command is preempted by a pre-start or mid-turn interrupt tombstone."""

    def __init__(self, conversation_id: str, turn_id: str) -> None:
        super().__init__(
            f"Turn '{turn_id}' for conversation '{conversation_id}' was preempted by cancel signal"
        )
        self.conversation_id = conversation_id
        self.turn_id = turn_id


class TokenBudgetSnapshot(BaseModel):
    """Serializable snapshot of session token consumption carried across stateless worker turns."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_tokens: int | None = None
    consumed_prompt_tokens: int = Field(default=0, ge=0)
    consumed_completion_tokens: int = Field(default=0, ge=0)
    per_provider: dict[str, int] = Field(default_factory=dict[str, int])
    turn_count: int = Field(default=0, ge=0)

    @property
    def total_consumed_tokens(self) -> int:
        """Total tokens consumed across prompt and completion."""
        return self.consumed_prompt_tokens + self.consumed_completion_tokens


#: Longest id a command may carry. Centrifugo refuses a channel name past 255 bytes, and a
#: session's channel is `ucx:session:<id>`.
MAX_WIRE_ID_LENGTH = 200


def wire_text_problem(value: str) -> str | None:
    """Why `value` cannot cross PostgreSQL, Valkey and JSON safely, or `None` if it can.

    A NUL makes PostgreSQL refuse the text, and a lone surrogate cannot be encoded as UTF-8 at
    all, so a record holding either fails the same way on every worker, for ever.
    """
    if "\x00" in value:
        return "contains a NUL"
    if any("\ud800" <= ch <= "\udfff" for ch in value):
        return "contains a lone surrogate"
    return None


def wire_id_problem(value: str) -> str | None:
    """Why `value` cannot be an id on the wire, or `None` if it can."""
    if not value.strip():
        return "is blank"
    if len(value) > MAX_WIRE_ID_LENGTH:
        return f"is longer than {MAX_WIRE_ID_LENGTH} characters"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value.replace("\x00", "")):
        return "contains a control character"
    return wire_text_problem(value)


#: How the ids a worker derives begin: a seat turn's, a landing's and a learn job's.
SEAT_TURN_ID_PREFIX: Final[str] = "seat-"
LANDED_ID_PREFIX: Final[str] = "landed-"
LEARN_JOB_ID_PREFIX: Final[str] = "learn-"


def is_worker_id(value: str) -> bool:
    """Whether `value` has the form of an id a worker derives, which a route refuses from a client.

    A receipt is known by its command's id, and these can be computed from a post's id: a client
    that took one would answer for the command the worker is yet to send.
    """
    return value.startswith((SEAT_TURN_ID_PREFIX, LANDED_ID_PREFIX, LEARN_JOB_ID_PREFIX))


class SeatTurnRef(BaseModel):
    """Marks a command on `ucx.cmd.turn` as a `SEAT_TURN_EXECUTE`: one speaker's turn in a room."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    room_id: str
    cascade_id: str

    @field_validator("room_id", "cascade_id")
    @classmethod
    def _valid_identifier(cls, value: str) -> str:
        problem = wire_id_problem(value)
        if problem is not None:
            raise ValueError(f"An identifier on SeatTurnRef {problem}")
        return value


class TurnCommandEnvelope(BaseModel):
    """Payload published to `ucx.cmd.turn` (partitioned by `session_id`).

    A seat's turn (`seat`) has an empty `prompt`: the worker that runs it renders the span the
    seat has not seen from the room, once it has the seat's own context window to fit it to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    turn_id: str
    session_id: str
    agent_id: str
    user_id: str
    prompt: str
    expected_revision: int | None = None
    #: Continue the turn its usage ceiling stopped (#2431), in place: `prompt` is not appended
    #: again, and the turn is not checked against the limit at its start.
    continues_stopped_turn: bool = False
    seat: SeatTurnRef | None = None
    traceparent: str | None = None
    enqueued_at_ms: int = Field(ge=0)

    @field_validator("turn_id", "session_id", "agent_id", "user_id")
    @classmethod
    def _valid_identifier(cls, value: str) -> str:
        problem = wire_id_problem(value)
        if problem is not None:
            raise ValueError(f"An identifier on TurnCommandEnvelope {problem}")
        return value

    @field_validator("prompt")
    @classmethod
    def _storable_prompt(cls, value: str) -> str:
        problem = wire_text_problem(value)
        if problem is not None:
            raise ValueError(f"The prompt on TurnCommandEnvelope {problem}")
        return value


class RoomCommandEnvelope(BaseModel):
    """Payload published to `ucx.cmd.room` (partitioned by `room_id`).

    `turn_id` names the command and is its receipt's key, so a redelivered copy is known by it.
    What the room's next step is comes from the room's own state, never from this record: a
    copy cannot move the turn ceiling or name a speaker. A `ROOM_TURN_LANDED` points at the seat
    turn's receipt instead of carrying the reply, so a reply of any size fits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["ROOM_POST", "ROOM_TURN_LANDED", "ROOM_INTERRUPT"]
    turn_id: str
    room_id: str
    user_id: str
    cascade_id: str
    #: `ROOM_POST`: the room's human who said it, and what they said.
    sender_id: str | None = None
    human_utterance: str | None = None
    #: `ROOM_TURN_LANDED`: the seat turn that ended.
    seat_turn_id: str | None = None
    traceparent: str | None = None
    enqueued_at_ms: int = Field(ge=0)

    @field_validator("turn_id", "room_id", "user_id", "cascade_id")
    @classmethod
    def _valid_identifier(cls, value: str) -> str:
        problem = wire_id_problem(value)
        if problem is not None:
            raise ValueError(f"An identifier on RoomCommandEnvelope {problem}")
        return value

    @field_validator("human_utterance")
    @classmethod
    def _storable_utterance(cls, value: str | None) -> str | None:
        problem = wire_text_problem(value) if value is not None else None
        if problem is not None:
            raise ValueError(f"The utterance on RoomCommandEnvelope {problem}")
        return value


class LearnCommandEnvelope(BaseModel):
    """`ROOM_LEARN_JOB`, published to `ucx.bg.learn` (partitioned by `room_id`).

    The rows a finished cascade landed, `from_seq` to `up_to_seq`; `job_id` is the job's receipt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    job_id: str
    room_id: str
    user_id: str
    from_seq: int = Field(ge=0)
    up_to_seq: int = Field(ge=0)
    traceparent: str | None = None


class RoomStepOutbox(BaseModel):
    """What one room step sends once it has committed, stored in the step's receipt.

    The receipt holds it for the case of a commit that succeeded and a publish that did not: a
    redelivered command finds the receipt, changes nothing, and sends this again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    events: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    seat_turns: list[TurnCommandEnvelope] = Field(default_factory=list[TurnCommandEnvelope])
    learn_jobs: list[LearnCommandEnvelope] = Field(default_factory=list[LearnCommandEnvelope])


class InterruptControlEnvelope(BaseModel):
    """Payload broadcast on `ucx.ctrl.interrupt` to preempt active or queued turns."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    conversation_id: str
    turn_id: str | None = None
    requested_by: str
    issued_at_ms: int = Field(ge=0)

    @field_validator("conversation_id", "turn_id")
    @classmethod
    def _valid_identifier(cls, value: str | None) -> str | None:
        problem = wire_id_problem(value) if value is not None else None
        if problem is not None:
            raise ValueError(f"An identifier on InterruptControlEnvelope {problem}")
        return value


class DeadLetterEnvelope(BaseModel):
    """A command given up on, as published to `ucx.dlq.turn`.

    It always fits in a Kafka record, so it describes the value instead of carrying it: the
    source's topic, partition and offset say where the whole record stays until retention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_topic: str
    source_partition: int | None = None
    source_offset: int | None = None
    #: The record's key, cut to a size that fits.
    key: str
    #: The length in bytes and the SHA-256 of the record's value as it was consumed.
    value_size: int = Field(ge=0)
    value_sha256: str
    #: The first bytes of that value, base64, because it may not be valid JSON.
    value_head_base64: str
    #: `malformed`, `attempts_used`, `deadline_passed`, `started_repeatedly` or `kept_changing`.
    reason: str
    #: The class of the last exception, when there was one. Its text stays in the log.
    error_kind: str | None = None
    attempts: int = Field(default=0, ge=0)
    starts: int = Field(default=0, ge=0)
    worker_id: str
    dead_lettered_at_ms: int = Field(ge=0)


class SSEStreamEnvelope(BaseModel):
    """Centrifugo-compatible publication envelope emitted to `ucx.stream.sse`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    method: Literal["publish"] = "publish"
    channel: str
    idempotency_key: str
    data: Mapping[str, Any]

    @field_validator("channel")
    @classmethod
    def _validate_channel_namespace(cls, value: str) -> str:
        if not value.startswith(("ucx:session:", "ucx:room:")):
            raise ValueError(
                f"SSEStreamEnvelope channel must start with 'ucx:session:' or 'ucx:room:', got {value!r}"
            )
        return value


class TurnLeaseToken(BaseModel):
    """Fenced turn execution lease held by a worker while executing a turn."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    conversation_id: str
    turn_id: str
    worker_id: str
    fencing_token: int = Field(ge=1)
    ttl_seconds: int = Field(default=60, ge=1)


@runtime_checkable
class CacheClientProtocol(Protocol):
    """Async client for the Valkey cluster holding hot cache and coordination state.

    The cluster runs `maxmemory-policy volatile-lru`, which evicts only keys that carry a TTL,
    so the two key classes are kept apart by the methods that write them:

    * **Cache keys** (`sess:head`, `blob:*`, `room:head`, `catalog:*`) always carry a TTL:
      `set_cache` / `add_cache` refuse to write one without, and `VERSIONED_HEAD_SET_SCRIPT`
      sets `EX`. They may be evicted at any time; PostgreSQL is the source of truth they are
      re-read from.
    * **Coordination keys** (lease, fencing, cancel tombstones, SSE sequence, usage windows)
      are written only through `incr` and `run_script`, and never carry a TTL, so eviction can
      never remove them. Expiry that coordination needs (a lease deadline, a tombstone's
      lifetime) is stored in the value and checked against the server clock inside a script.

    Every key one script touches must share a hash tag (`{conversation_id}`) so that the keys
    live in one cluster slot and the script runs atomically on one node.
    """

    async def get(self, key: bytes) -> bytes | None: ...
    async def multi_get(self, *keys: bytes) -> tuple[bytes | None, ...]: ...
    async def set_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> None: ...
    async def add_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> bool: ...
    async def delete(self, *keys: bytes) -> int: ...
    async def hash_get_all(self, key: bytes) -> Mapping[bytes, bytes]: ...
    async def incr(self, key: bytes, amount: int = 1) -> int: ...
    async def run_script(
        self, source: str, keys: Sequence[bytes], args: Sequence[bytes | str | int]
    ) -> Any: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


#: Cache write for a revisioned head (`sess:head:{id}`, `room:head:{id}`): stores
#: `"<revision>\n<json>"` only when no newer revision is cached, so a reader that loaded an
#: older row from PostgreSQL can never overwrite a newer head (Invariant 1). KEYS[1] = head key;
#: ARGV = revision, body, ttl seconds. Returns 1 when written, 0 when a newer head was kept.
VERSIONED_HEAD_SET_SCRIPT = """
local cur = redis.call('GET', KEYS[1])
if cur then
  local rev = tonumber(string.match(cur, '^(%d+)\\n'))
  if rev and rev >= tonumber(ARGV[1]) then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[1] .. '\\n' .. ARGV[2], 'EX', ARGV[3])
return 1
"""


def encode_versioned_head(revision: int, body: str) -> bytes:
    """Encode a head the way `VERSIONED_HEAD_SET_SCRIPT` stores it."""
    return f"{revision}\n{body}".encode()


def decode_versioned_head(raw: bytes) -> tuple[int, str]:
    """Split a cached head into `(revision, body)`."""
    text = raw.decode("utf-8")
    revision, sep, body = text.partition("\n")
    if not sep or not revision.isdigit():
        raise ValueError("cached head is not '<revision>\\n<body>'")
    return int(revision), body


@runtime_checkable
class SqlConnProtocol(Protocol):
    """Single async PostgreSQL connection and transaction handle."""

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None: ...
    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...
    async def execute(self, query: str, *args: Any) -> str: ...
    async def executemany(self, command: str, args: Sequence[Sequence[Any]]) -> None: ...
    def transaction(self) -> AbstractAsyncContextManager[None]: ...


@runtime_checkable
class SqlPoolProtocol(Protocol):
    """Async PostgreSQL connection pool contract."""

    def acquire(self) -> AbstractAsyncContextManager[SqlConnProtocol]: ...
    async def close(self) -> None: ...


@runtime_checkable
class ClusterSessionStoreProtocol(SessionStoreProtocol, Protocol):
    """Extends `SessionStoreProtocol` with two-phase blob prefetch (#1848) and budget snapshots."""

    async def prefetch_session(self, session_id: str) -> SessionState | None:
        """Two-phase `MC.multi_get` with PostgreSQL Read-Through fallback before turn start."""
        ...

    def evict_local_turn_cache(self, session_id: str) -> None:
        """Discard worker-local prefetched buffers after the turn finishes."""
        ...

    def save_budget_snapshot(
        self, session_id: str, snapshot: TokenBudgetSnapshot, *, expected_revision: int
    ) -> None:
        """Persist the snapshot iff the session record is still at `expected_revision`.

        Raises:
            StaleSessionWriteError: The record has moved past `expected_revision`.
        """
        ...

    async def settled_terminal(self, session_id: str, turn_id: str) -> Mapping[str, Any] | None:
        """The terminal event `turn_id`'s receipt holds, or `None` with no receipt (redelivery)."""
        ...

    async def settle_turn(self, session_id: str, turn_id: str, terminal: Mapping[str, Any]) -> None:
        """Write the receipt of a turn that ends without a commit, before anyone is told so."""
        ...

    async def record_turn_start(self, session_id: str, turn_id: str) -> int:
        """Count one start of this delivery, durably; returns the starts so far."""
        ...

    async def refund_turn_start(self, session_id: str, turn_id: str) -> None:
        """Take back a start that came to an end: only a process that died keeps one."""
        ...

    def commit_turn(
        self,
        state: SessionState,
        *,
        turn_id: str,
        terminal: Mapping[str, Any],
        budget: TokenBudgetSnapshot | None = None,
        pending_events: Sequence[Any] | None = None,
    ) -> SessionState:
        """Save `state` with `turn_id`'s receipt, `terminal` and `budget`, in one OCC transaction.

        Raises:
            StaleSessionWriteError: The record has moved past `state.revision`; nothing of
                the turn was written.
        """
        ...

    async def invalidate_head(self, session_id: str) -> None:
        """Drop the cached head, so the next prefetch reads the source of truth."""
        ...

    def load_budget_snapshot(self, session_id: str) -> TokenBudgetSnapshot | None:
        """Load the stored `TokenBudgetSnapshot` for `session_id`, if any."""
        ...


@runtime_checkable
class RoomStoreProtocol(Protocol):
    """Backend-neutral room persistence seam."""

    def load(self, room_id: str) -> RoomState | None: ...
    def save(self, state: RoomState) -> RoomState: ...
    def delete(self, room_id: str) -> bool: ...
    def list_room_ids(self) -> tuple[str, ...]: ...


@runtime_checkable
class ClusterRoomStoreProtocol(RoomStoreProtocol, Protocol):
    """A room store for workers on one event loop: a step's commit carries its receipt."""

    async def read_room(self, room_id: str) -> RoomState | None:
        """The room as the source of truth holds it, never the cached head.

        A worker that committed and died before writing the head leaves the head behind, and
        a floor read from it would look free.
        """
        ...

    async def step_receipt(self, room_id: str, command_id: str) -> RoomStepOutbox | None:
        """What `command_id`'s commit left to send, or `None` when it has not committed."""
        ...

    async def commit_step(
        self, state: RoomState | None, *, room_id: str, command_id: str, outbox: RoomStepOutbox
    ) -> RoomState | None:
        """Save `state` (if any) and the step's receipt in one transaction under the revision check.

        Raises:
            StaleRoomWriteError: The record moved past `state.revision`, or another worker
                committed this command; nothing was written.
        """
        ...


@runtime_checkable
class TurnLeaseProtocol(Protocol):
    """Fenced turn execution lock and pre-start interrupt tombstone coordinator."""

    async def acquire(
        self, conversation_id: str, turn_id: str, worker_id: str, ttl_seconds: int | None = None
    ) -> TurnLeaseToken: ...
    async def renew(self, token: TurnLeaseToken) -> bool: ...
    async def release(self, token: TurnLeaseToken) -> None: ...
    async def write_cancel_tombstone(
        self, conversation_id: str, turn_id: str, ttl_seconds: int = 300
    ) -> None: ...
    async def is_cancelled(self, conversation_id: str, turn_id: str) -> bool: ...


@runtime_checkable
class CommandBrokerProtocol(Protocol):
    """Producer and offset-commit abstraction over the distributed event backbone."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def publish_turn(self, cmd: TurnCommandEnvelope) -> int | None: ...
    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> int | None: ...
    async def publish_learn_job(self, cmd: LearnCommandEnvelope) -> int | None: ...
    async def broadcast_interrupt(self, ctrl: InterruptControlEnvelope) -> None: ...
    async def publish_sse(self, conversation_id: str, envelope: SSEStreamEnvelope) -> None: ...
    async def publish_dead_letter(self, envelope: DeadLetterEnvelope) -> None: ...
    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None: ...


@runtime_checkable
class SSEStreamPublisherProtocol(Protocol):
    """Bridges an in-process `EventBusProtocol` to the `ucx.stream.sse` topic."""

    async def publish_event(self, conversation_id: str, event: AgentEvent) -> int: ...
    def attach_to_bus(
        self, bus: EventBusProtocol, conversation_id: str
    ) -> AbstractAsyncContextManager[None]: ...


@runtime_checkable
class DistributedUsageStoreProtocol(Protocol):
    """Cluster-wide paid-token windows, and the durable ledger of every completed model call."""

    async def record_completed_call(
        self,
        *,
        scope: str,
        session_id: str,
        turn_id: str,
        entry: UsageEntry,
        counts_against_limits: bool,
    ) -> None: ...
    async def entries_since(self, scope: str, since: datetime) -> Sequence[UsageEntry]: ...
