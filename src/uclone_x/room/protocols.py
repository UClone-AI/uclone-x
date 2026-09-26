"""The seams of a multi-agent room: who speaks, where the room is kept, who runs the turn.

Interface-first, for the reason `core/session_store.py` gives: a protocol written after
its consumers is a protocol shaped by the first implementation that happened to exist.

`@runtime_checkable` is applied nowhere here, matching `agent/protocols.py`: no
`isinstance` check is performed against these, and conformance is enforced statically by
bindings in the protocol-conformance test.
"""

from __future__ import annotations

from typing import Protocol

from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.room.models import (
    Participant,
    RoomState,
    SpeakerDecision,
    SpeakerRequest,
)

__all__ = [
    "RoomAgentResolverProtocol",
    "RoomOrchestratorProtocol",
    "RoomStoreProtocol",
    "SpeakerSelectorProtocol",
    "StoryLeaseProtocol",
]


class SpeakerSelectorProtocol(Protocol):
    """Decides who speaks next. The room's one interesting decision.

    **A selector answers; it does not act.** It returns a `SpeakerDecision` and notifies
    nobody, so it is a pure function of its `SpeakerRequest` and can be tested without a
    live agent, an event bus or a model.

    **Three answers, and a fourth that is not an answer.** `SPEAK` names a speaker,
    `SILENCE` decides that nobody should, `ABSTAIN` declines and passes to the next
    selector in the chain. Failure is *none of these*: a selector whose model call raises,
    whose response will not parse, or whose configuration is absent **must raise**. It must
    not return `SILENCE` (which claims a judgement it did not make) and must not return
    `ABSTAIN` (which hides a broken selector behind whichever one runs next). This is P6
    at the seam: a substituted default result is forbidden, and "nobody spoke" must not be
    the shape both a working quiet room and a dead selector take.

    **Cheap first.** Implementations are chained cheapest-first — a rule selector that
    answers a direct address or a one-agent room without a model call, then a selector
    that spends one. A chain that abstains throughout is resolved to silence by the
    orchestrator, which is the one place that conversion is allowed to happen.

    **A named speaker is not yet a valid one.** `speaker_id` is checked against the live
    roster by the orchestrator, not here: a selector naming a participant that has left is
    an ordinary event in a room whose roster changes, and belongs in the decision trail
    rather than in an exception.
    """

    @property
    def name(self) -> str:
        """Stable identifier, recorded in `SpeakerDecision.selector` and the transcript."""
        ...

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        """Return who should speak next.

        Async because an implementation may call a model; a rule-based one returns without
        awaiting and costs nothing beyond the coroutine.

        Raises:
            SpeakerSelectionError: The selector could not reach a judgement — a provider
                failure, an unparseable response, or missing configuration. Never returned
                as a verdict.
        """
        ...


class RoomAgentResolverProtocol(Protocol):
    """Supplies the live agent behind a participant, keeping each one's state its own.

    The room needs agents but must not own their lifecycle: the head already does (P8
    keeps business logic out of the head, and this protocol is what lets the room stay in
    the Core while the head keeps its agent registry). The room holds this seam and
    nothing else of the agent world.

    **The isolation obligation is on the implementation, and it is not incidental.** For
    two distinct participants the resolver must return agents that share no session and no
    ontology: each agent is constructed against its participant's own `session_id` and its
    own ontology namespace (P7 — an agent's knowledge graph is built from *its* experience
    and merging two of them destroys the grounding), and returning one agent for two
    participants, or two agents pointed at one session, puts two writers on one
    `SessionState` and is refused downstream by the store's revision precondition. Tool
    registries and skill sets may be shared; the session and the ontology may not.
    """

    async def resolve(self, participant: Participant) -> BaseAgentProtocol:
        """Return the live agent for `participant`, constructing it if needed.

        Raises:
            ParticipantNotResolvableError: No agent can be produced for this participant —
                an unknown agent id, or a human participant, which has no agent.
        """
        ...


class RoomStoreProtocol(Protocol):
    """Durable storage for rooms. The kernel owns *when*, the host owns *where*.

    Deliberately the same shape as `SessionStoreProtocol`, including the compare-and-swap
    on `save`, so that a host implementing both does not meet two different persistence
    vocabularies.
    """

    def load(self, room_id: str) -> RoomState | None:
        """Return the stored room, or `None` when there is none.

        A backend that can tell "no such room" from "storage unreachable" must raise for
        the second: returning `None` there makes the caller seed an empty room and
        overwrite a conversation that was merely unreachable.

        A record that is there and will not load raises `UnreadableRoomRecordError`, not
        the parser's own exception: that one reads to a caller as a malformed request.
        """
        ...

    def save(self, state: RoomState) -> RoomState:
        """Persist `state` and return what a subsequent `load` will yield.

        Refuses when the stored record has moved past `state.revision`, so a lost update
        is an error rather than a silent overwrite. The returned state carries the
        incremented revision.
        """
        ...

    def delete(self, room_id: str) -> bool:
        """Remove the room; return whether one was there to remove."""
        ...

    def list_room_ids(self) -> tuple[str, ...]:
        """Every stored room id."""
        ...


class RoomOrchestratorProtocol(Protocol):
    """Runs the room: takes an utterance, and drives turns until the floor is quiet.

    It is the room's **single writer** — the property the store's precondition exists to
    protect — and the only component permitted to convert an all-abstaining selector chain
    into recorded silence.
    """

    async def post(self, room_id: str, sender_id: str, content: str) -> RoomState:
        """Append an utterance and drive the resulting agent turns to a stop.

        Stops at the first `SILENCE`, at an exhausted selector chain, or when
        `RoomPolicy.max_agent_turns_per_human_message` is spent — whichever comes first.
        Each agent turn appends *only* the speaker's final utterance to the transcript;
        its tool calls and intermediate turns stay in that agent's own session.

        A turn that fails is recorded on the transcript with `RoomMessage.error` set and
        does not advance that participant's `last_seen_seq`, so the failure is visible and
        the unseen span is not silently consumed.

        Returns:
            The room as persisted after the last append.
        """
        ...

    async def accept(self, room_id: str, sender_id: str, content: str) -> RoomState:
        """Record a human utterance and return, giving nobody the floor.

        The half of `post` a request can await. Makes every refusal `post` makes, so an
        unknown sender is still an answer to the caller rather than a later event.
        """
        ...

    async def resume(self, room_id: str, baseline_seq: int) -> RoomState:
        """Drive the turns for an utterance already recorded at `baseline_seq`."""
        ...

    async def retry(self, room_id: str) -> RoomState:
        """Re-run the room's last utterance when it was a failed turn.

        The floor goes back to the agent that failed, under the decision that gave it the
        floor, and **reuses the turn slot the failure spent** rather than taking a fresh
        one from `RoomPolicy.max_agent_turns_per_human_message`: the failed turn produced
        no conversation, and the budget bounds conversation. The failed row is kept; the
        retry appends beside it.

        Raises:
            NothingToRetryError: The last utterance did not fail.
            UnknownRoomParticipantError: The agent that failed has left the room.
        """
        ...

    async def note_human_activity(self, room_id: str, sender_id: str = "human") -> None:
        """Record that a human is active — typing, not yet sent.

        Advances `TurnState.last_activity_ts`, which is what a hesitating turn compares
        across its wait in order to yield the floor. A head that cannot report typing
        simply never calls this, and hesitation degrades to a plain delay — which is why
        `RoomPolicy.hesitation_seconds` defaults to off.
        """
        ...

    async def interrupt(self, room_id: str, reason: str = "Turn was interrupted") -> None:
        """Signal an interrupt/stop to a running room.

        Cancels any currently executing agent turn for `room_id`, records the interrupted
        turn in the transcript with `completed=False` and an error notice, and terminates
        the active turn loop.
        """
        ...


class StoryLeaseProtocol(Protocol):
    """Gives back a story's writing lease when the conversation holding it is deleted.

    The room kernel names only this, not the story library, which is an adapter: the
    head that builds a `RoomService` passes the library, and `StoryLibrary` has this shape.
    """

    def release(self, story_id: str, conversation_id: str) -> bool:
        """Give up the lease if `conversation_id` holds it; return whether it did."""
        ...
