"""Stateless API ingress service for command submission and dual-path turn interruption."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from uclone_x.core.cluster_protocols import (
    CommandBrokerProtocol,
    InterruptControlEnvelope,
    RoomCommandEnvelope,
    TurnCommandEnvelope,
    TurnLeaseProtocol,
)

__all__ = ["ClusterIngressService", "IngressAcceptedResponse"]


class IngressAcceptedResponse(BaseModel):
    """202 Accepted response returned by stateless ingress endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    status_code: int = 202
    turn_id: str
    conversation_id: str
    sse_channel: str


class ClusterIngressService:
    """Stateless API gateway coordinator for turn submission, room steps, and interrupts."""

    def __init__(
        self,
        broker: CommandBrokerProtocol,
        lease_manager: TurnLeaseProtocol,
    ) -> None:
        self._broker = broker
        self._lease_manager = lease_manager

    async def submit_turn(self, cmd: TurnCommandEnvelope) -> IngressAcceptedResponse:
        """Enqueue a single-agent turn command and return a 202 Accepted SSE subscription descriptor."""
        await self._broker.publish_turn(cmd)
        return IngressAcceptedResponse(
            status_code=202,
            turn_id=cmd.turn_id,
            conversation_id=cmd.session_id,
            sse_channel=f"ucx:session:{cmd.session_id}",
        )

    async def submit_room_message(self, cmd: RoomCommandEnvelope) -> IngressAcceptedResponse:
        """Enqueue a person's `ROOM_POST` and return a 202 Accepted SSE subscription descriptor."""
        await self._broker.publish_room_step(cmd)
        return IngressAcceptedResponse(
            status_code=202,
            turn_id=cmd.turn_id,
            conversation_id=cmd.room_id,
            sse_channel=f"ucx:room:{cmd.room_id}",
        )

    async def interrupt_room(
        self, *, room_id: str, user_id: str, cascade_id: str, requested_by: str, command_id: str
    ) -> None:
        """Stop a room: the seat on its floor, queued or running, and what would follow it.

        `ROOM_INTERRUPT` goes in the room's own order, so it stops whichever seat holds the floor
        when it is reached, and marks the room so that seat's landing starts nobody for what was
        said by then. The broadcast then reaches a seat running now without waiting for the
        room's queue.
        """
        now = int(time.time() * 1000)
        await self._broker.publish_room_step(
            RoomCommandEnvelope(
                kind="ROOM_INTERRUPT",
                turn_id=command_id,
                room_id=room_id,
                user_id=user_id,
                cascade_id=cascade_id,
                enqueued_at_ms=now,
            )
        )
        await self._broker.broadcast_interrupt(
            InterruptControlEnvelope(
                conversation_id=room_id, requested_by=requested_by, issued_at_ms=now
            )
        )

    async def interrupt_turn(self, ctrl: InterruptControlEnvelope) -> None:
        """Tombstone the named turn, then broadcast the interrupt to every worker.

        The tombstone comes first, so a worker that pops the turn after this returns refuses it
        whether or not it saw the broadcast. An interrupt naming no turn has nothing to
        tombstone: it stops what is running on the conversation now, and only that.
        """
        if ctrl.turn_id is not None:
            await self._lease_manager.write_cancel_tombstone(ctrl.conversation_id, ctrl.turn_id)
        await self._broker.broadcast_interrupt(ctrl)
