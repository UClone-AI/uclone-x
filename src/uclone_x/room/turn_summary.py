"""Summarize one turn's tool use, changed documents, and selection provenance (#1491).

Provides the Core read model for the user-facing Turn surface in the workspace dock.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.provenance import Provenance
from uclone_x.errors import RoomError
from uclone_x.llm.models import TokenUsage
from uclone_x.room.models import (
    ParticipantKind,
    RoomMessage,
    RoomState,
    RoomToolUse,
    RoomTurnRefusal,
    SpeakerDecision,
)

__all__ = [
    "TurnDocument",
    "TurnNotFoundError",
    "TurnSummary",
    "summarize_turn",
]


class TurnNotFoundError(RoomError):
    """Raised when summarizing a turn sequence number not in the room transcript."""

    def __init__(self, room_id: str, seq: int) -> None:
        self.room_id = room_id
        self.seq = seq
        super().__init__(f"Turn {seq} was not found in conversation {room_id}.")


class TurnDocument(BaseModel):
    """A distinct document written during a turn and how many writes it received."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str
    writes: int


class TurnSummary(BaseModel):
    """Summary of what one turn did for general user inspection."""

    model_config = ConfigDict(extra="ignore")

    seq: int
    turn_id: str | None = None
    sender_id: str
    created_at: str | None = None
    completed: bool = True
    error: str | None = None
    refusal: RoomTurnRefusal | str | None = None
    provenance: Provenance | None = None
    decision: SpeakerDecision | None = None
    rendered_through: int = 0
    usage: TokenUsage | None = None
    notable: list[str] = Field(default_factory=list[str])
    steps: list[RoomToolUse] = Field(default_factory=list[RoomToolUse])
    documents: list[TurnDocument] = Field(default_factory=list[TurnDocument])
    unnamed_writes: int = 0
    subagent_steps: list[RoomToolUse] = Field(default_factory=list[RoomToolUse])
    steps_absent_reason: str | None = None


def _compute_notable(message: RoomMessage) -> list[str]:
    notable: list[str] = []
    prov = message.provenance
    if prov is not None:
        if getattr(prov, "degraded", False):
            notable.append("degraded")
        path_val = getattr(prov, "path", None)
        if path_val is not None:
            path_str = str(getattr(path_val, "value", path_val))
            if path_str == "failover":
                notable.append("failover")
            elif path_str == "retry":
                notable.append("retried")

    dec = message.decision
    if dec is not None:
        selector = getattr(dec, "selector", "")
        candidates = getattr(dec, "candidates", None)
        is_contested = (selector != "sole_agent") or (
            candidates is not None and len(candidates) > 1
        )
        if is_contested:
            notable.append("contested")

    return notable


def summarize_turn(state: RoomState, seq: int) -> TurnSummary:
    """Summarize what one turn did from the room state."""
    message = next((m for m in state.transcript if m.seq == seq), None)
    if message is None:
        raise TurnNotFoundError(state.room_id, seq)

    participant = next((p for p in state.participants if p.id == message.sender_id), None)
    is_agent = participant is not None and participant.kind == ParticipantKind.AGENT

    notable = _compute_notable(message)

    if not is_agent:
        return TurnSummary(
            seq=message.seq,
            turn_id=message.turn_id,
            sender_id=message.sender_id,
            created_at=message.created_at,
            completed=message.completed,
            error=message.error,
            refusal=message.refusal,
            provenance=message.provenance,
            decision=message.decision,
            rendered_through=message.rendered_through,
            usage=message.usage,
            notable=notable,
            steps=[],
            documents=[],
            unnamed_writes=0,
            subagent_steps=[],
            steps_absent_reason="not_an_agent_turn",
        )

    if not message.tools_recorded:
        return TurnSummary(
            seq=message.seq,
            turn_id=message.turn_id,
            sender_id=message.sender_id,
            created_at=message.created_at,
            completed=message.completed,
            error=message.error,
            refusal=message.refusal,
            provenance=message.provenance,
            decision=message.decision,
            rendered_through=message.rendered_through,
            usage=message.usage,
            notable=notable,
            steps=[],
            documents=[],
            unnamed_writes=0,
            subagent_steps=[],
            steps_absent_reason="not_recorded",
        )

    turn_id = message.turn_id or ""
    turn_uses = [u for u in state.tool_uses if u.turn_id == turn_id]
    steps = sorted(turn_uses, key=lambda u: u.recorded_at)

    docs_map: dict[str, int] = {}
    for u in steps:
        if u.written_path:
            docs_map[u.written_path] = docs_map.get(u.written_path, 0) + 1
    documents = [TurnDocument(path=p, writes=c) for p, c in docs_map.items()]

    unnamed_writes = sum(1 for u in steps if u.wrote_unnamed)
    subagent_steps = [u for u in steps if u.subagent_id is not None]

    return TurnSummary(
        seq=message.seq,
        turn_id=message.turn_id,
        sender_id=message.sender_id,
        created_at=message.created_at,
        completed=message.completed,
        error=message.error,
        refusal=message.refusal,
        provenance=message.provenance,
        decision=message.decision,
        rendered_through=message.rendered_through,
        usage=message.usage,
        notable=notable,
        steps=steps,
        documents=documents,
        unnamed_writes=unnamed_writes,
        subagent_steps=subagent_steps,
        steps_absent_reason=None,
    )
