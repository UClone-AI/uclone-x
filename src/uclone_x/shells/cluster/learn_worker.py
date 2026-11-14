"""The `ucx.bg.learn` consumer: what a finished cascade taught its clones, off the critical path.

`ROOM_LEARN_JOB` names the rows a cascade landed. For each clone that spoke in them and still
learns (#1840), this runs the same `KnowledgeExtractor` a room runs on one machine, and notes the
outcome on the clone's row through the room's revision check, so a person's reply committing
meanwhile is kept and the note goes on top of it.

**Where the facts go is the open part.** `Lesson.memory` needs a `FactStoreProtocol`, and a worker
has none: `CrossSessionMemory` is a file per clone, and `PostgresKnowledgeStore` only upserts and
lists bare triples, with no fact ids, provenance, reinforcement or supersession, and no owner. So
`ucx worker` passes no `fact_store_for`, and then a job is acknowledged with no model call: asking
a model for facts that nothing can keep spends the person's money for nothing.

**Nothing here can touch the room's cascade.** It has a consumer and a dispatcher of its own, so a
slow lesson holds no turn's slot and no room's offset, and it never raises for a failure to learn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Final

from uclone_x.core.cluster_protocols import (
    ClusterRoomStoreProtocol,
    LearnCommandEnvelope,
    RoomStepOutbox,
)
from uclone_x.errors import StaleRoomWriteError
from uclone_x.memory.extractor import (
    EXTRACTION_FAILED,
    ExtractionOutcome,
    FactStoreProtocol,
    KnowledgeExtractor,
    Lesson,
)
from uclone_x.room.models import Participant, ParticipantKind, RoomState
from uclone_x.room.orchestrator import (
    clone_names_in,
    landed_lesson,
    person_names_in,
    with_learning_recorded,
)
from uclone_x.shells.cluster.turn_executor import AgentTurnExecutor

__all__ = ["FactStoreFor", "RoomLearnWorker"]

logger = logging.getLogger(__name__)

#: Lost races with a person's reply before a note is dropped; the facts stay saved either way.
_MAX_NOTE_ATTEMPTS: Final[int] = 5

#: Where a clone's facts go, for `(clone id, user id)`.
FactStoreFor = Callable[[str, str], FactStoreProtocol | None]


class RoomLearnWorker:
    """Runs `ROOM_LEARN_JOB`s; a job's receipt makes a redelivered one a no-op."""

    def __init__(
        self,
        *,
        rooms: ClusterRoomStoreProtocol,
        clones: AgentTurnExecutor,
        fact_store_for: FactStoreFor | None = None,
    ) -> None:
        self._rooms = rooms
        self._clones = clones
        self._fact_store_for = fact_store_for
        self._extractor = KnowledgeExtractor()

    async def process_learn_job(self, job: LearnCommandEnvelope) -> None:
        """Learn what `job` names. A failure is logged and the job is over: nothing is retried."""
        try:
            await self._process(job)
        except Exception:
            logger.exception("Learn job %s failed; its room is untouched", job.job_id)

    async def _process(self, job: LearnCommandEnvelope) -> None:
        if await self._rooms.step_receipt(job.room_id, job.job_id) is not None:
            return
        state = await self._rooms.read_room(job.room_id)
        if state is None:
            return
        if self._fact_store_for is None:
            logger.info("Learn job %s: this worker has nowhere to keep facts", job.job_id)
            await self._acknowledge(job)
            return
        outcomes = [
            await self._learn(state, job, speaker)
            for speaker in state.participants
            if speaker.kind is ParticipantKind.AGENT
        ]
        await self._note(job, [outcome for outcome in outcomes if outcome is not None])

    async def _acknowledge(self, job: LearnCommandEnvelope) -> None:
        await self._rooms.commit_step(
            None, room_id=job.room_id, command_id=job.job_id, outbox=RoomStepOutbox()
        )

    async def _learn(
        self, state: RoomState, job: LearnCommandEnvelope, speaker: Participant
    ) -> ExtractionOutcome | None:
        assert self._fact_store_for is not None
        turn_ids, lines = landed_lesson(state, speaker, job.from_seq, job.up_to_seq)
        if not turn_ids:
            return None
        memory = self._fact_store_for(speaker.id, job.user_id)
        if memory is None:
            return None
        clone = self._clones.compose_learner(state, speaker, user_id=job.user_id, job_id=job.job_id)
        lesson = Lesson(
            clone_id=speaker.id,
            clone_name=speaker.display_name,
            room_id=state.room_id,
            session_id=speaker.session_id or state.room_id,
            turn_id=turn_ids[-1],
            turn_ids=turn_ids,
            lines=lines,
            memory=memory,
            generate=clone.agent.invoke_auxiliary_model,
            person_names=person_names_in(state),
            clone_names=clone_names_in(state, speaker),
            learns=lambda: self._clones.clone_learns(speaker.id),
        )
        try:
            return await self._extractor.extract(lesson)
        except Exception:  # extract returns its failures; this is a defect in it
            logger.exception("Learning from turn %s failed", lesson.turn_id)
            return ExtractionOutcome(
                clone_id=speaker.id,
                room_id=state.room_id,
                turn_id=lesson.turn_id,
                error=EXTRACTION_FAILED,
            )
        finally:
            await clone.llm.aclose()

    async def _note(self, job: LearnCommandEnvelope, outcomes: Sequence[ExtractionOutcome]) -> None:
        """Write what was learned on each row, in one commit with the job's receipt."""
        noted = [o for o in outcomes if o.added or o.error is not None or o.skipped]
        for _ in range(_MAX_NOTE_ATTEMPTS):
            state = await self._rooms.read_room(job.room_id)
            if state is None:
                return
            changed = state
            for outcome in noted:
                changed = with_learning_recorded(changed, outcome) or changed
            try:
                await self._rooms.commit_step(
                    changed if changed is not state else None,
                    room_id=job.room_id,
                    command_id=job.job_id,
                    outbox=RoomStepOutbox(),
                )
                return
            except StaleRoomWriteError:
                # Someone moved the room while this learned: read it again, never write over it.
                continue
        logger.warning("Learn job %s could not note its outcome", job.job_id)
