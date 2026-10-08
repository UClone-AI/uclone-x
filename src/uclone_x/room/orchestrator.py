"""The room's decision-maker: it runs the selector chain, and it alone gives the floor.

**The division of authority is the point.** A selector *judges* and returns a value; the
orchestrator *decides and acts*. Everything that is not judgement lives here: chain order,
validating a named speaker against the live roster, the turn ceiling, converting an
exhausted chain into recorded silence, yielding to a human, and writing the transcript.
A selector cannot do any of those, which is what makes every routing rule testable without
a room.

It is also the room's **single writer**, which is the property `RoomStore.save`'s
compare-and-swap exists to protect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from uclone_x.agent.models import ProviderFailure, ToolExecutionRecord
from uclone_x.agent.protocols import BaseAgentProtocol
from uclone_x.agent.session import SessionState
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.engine.event_bus import AgentEvent, EventSource, EventType
from uclone_x.engine.protocols import EventBusProtocol, PublisherHandleProtocol
from uclone_x.errors import (
    HeadRoomWriteError,
    NothingToRetryError,
    ProviderFailureKind,
    RoomError,
    RoomNotFoundError,
    SpeakerSelectionError,
    TurnNotLandedError,
    TurnNotStartedError,
    UnknownRoomParticipantError,
)
from uclone_x.llm.compactor import estimate_text_tokens
from uclone_x.memory.extractor import (
    EXCLUDED_TOOL_PREFIXES,
    EXTRACTION_FAILED,
    ExtractionOutcome,
    KnowledgeExtractor,
    LearningSeatProtocol,
    Lesson,
    SpanLine,
)
from uclone_x.memory.models import fold_name
from uclone_x.room.models import (
    SPAN_WINDOW_SHARE,
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomMessageKind,
    RoomState,
    RoomToolUse,
    RoomTurnRefusal,
    RoomWrittenFile,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
    head_room_write_refusal,
    is_one_seat,
    turn_refusal,
)
from uclone_x.room.protocols import (
    RoomAgentResolverProtocol,
    RoomStoreProtocol,
    SeatSessionSetAsideProtocol,
    SpeakerSelectorProtocol,
)
from uclone_x.tools.models import ToolResultStatus

__all__ = ["RoomOrchestrator", "memory_save_outcome", "record_turn_tools"]

logger = logging.getLogger(__name__)

#: The failures of a clone's own model ref that name it and offer the system default (§3.6):
#: its connection removed or not listing the model, or not answering.
_OWN_MODEL_FAILURES = frozenset(
    {ProviderFailureKind.MODEL_UNAVAILABLE, ProviderFailureKind.PROVIDER_UNREACHABLE}
)

#: Characters of a tool call's arguments, and of its result, the room keeps (#1353). The
#: full trace is the seat's own (G1, G2); the room's record is for a reader asking what a
#: seat did, and a `file_write`'s arguments carry the whole file.
TOOL_PREVIEW_CHARS = 2000

#: Circuit breaker ceiling for autonomous room discussions
AUTONOMOUS_CIRCUIT_BREAKER_TURNS: Final = 20
#: Default hesitation pause between turns in autonomous mode to ensure readable pacing
AUTONOMOUS_DEFAULT_HESITATION_SECONDS: Final = 1.5


def _preview(value: Any) -> tuple[str, bool]:
    """`value` as bounded text, and whether it had to be cut."""
    if value is None:
        return "", False
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= TOOL_PREVIEW_CHARS:
        return text, False
    return text[:TOOL_PREVIEW_CHARS], True


#: The name `memory/tools.py::RecordMemoryFactTool` registers under. Spelled here rather
#: than imported because the orchestrator is kernel and the memory tools are an adapter
#: (docs/core-shell-architecture.md §4); a unit test pins the two together.
MEMORY_SAVE_TOOL_NAME = "record_memory_fact"


def _memory_fact_key(arguments: Mapping[str, Any]) -> tuple[str, str, str]:
    """The `(subject, predicate, object_value)` a save names, normalised; `""` for a part it left out."""

    def part(name: str) -> str:
        value = arguments.get(name)
        return value.strip().casefold() if isinstance(value, str) else ""

    return part("subject"), part("predicate"), part("object_value")


def _agrees(named: tuple[str, str, str], other: tuple[str, str, str]) -> bool:
    """Whether `other` matches `named` on every part `named` actually names."""
    return all(mine == "" or mine == theirs for mine, theirs in zip(named, other, strict=True))


def _memory_save_outcome(executions: Sequence[ToolExecutionRecord]) -> tuple[int, int]:
    """`(tried, unsaved)`: the distinct facts a turn asked to save, and how many never were.

    A fact is identified by the `(subject, predicate, object_value)` its call named,
    compared after trimming and case-folding. A failed save counts as saved when any
    successful save in the same turn agrees with it on every part the failed call named,
    provided the failure named at least a subject or a predicate -- a failed save that
    names neither subject nor predicate is never resolved by a success. A retry that fixed
    a missing value, or supplied a subject the first call left out, resolves the failure.
    Failed saves that no success resolves are counted once per fact they can be told apart
    as: two failures where one's named parts agree with the other's are one fact. The
    counts are only as exact as those names are. Error text is never read, since it is
    written for the model and not as a reason a reader may be shown.
    """
    saves = [e for e in executions if e.tool_name == MEMORY_SAVE_TOOL_NAME]
    landed = {
        _memory_fact_key(save.arguments)
        for save in saves
        if save.status == ToolResultStatus.SUCCESS
    }
    unsaved: list[tuple[str, str, str]] = []
    for save in saves:
        if save.status == ToolResultStatus.SUCCESS:
            continue
        key = _memory_fact_key(save.arguments)
        has_identity = bool(key[0] or key[1])
        if has_identity and any(_agrees(key, done) for done in landed):
            continue
        if any(_agrees(key, seen) or _agrees(seen, key) for seen in unsaved):
            continue
        unsaved.append(key)
    return len(landed) + len(unsaved), len(unsaved)


#: The image tool, whose results carry the tags its fill added to each picture's prompts.
IMAGE_TOOL_NAME = "generate_image"


def _image_prompt_additions(
    executions: Sequence[ToolExecutionRecord],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(prompt, negative)`: the tags the image tool added to this turn's pictures (#1865).

    Read from each successful `generate_image` result -- the result itself, then each of
    its `images` for a batch, since a batch says tags every picture shares once at the top
    (#2013) -- in order, each tag once. Only lists of strings are read; the tool's
    `prompt_changes` sentences are written for the model and are not.
    """
    prompt: list[str] = []
    negative: list[str] = []

    def take(into: list[str], value: object) -> None:
        if isinstance(value, list):
            for tag in cast(list[object], value):
                if isinstance(tag, str) and tag and tag not in into:
                    into.append(tag)

    for execution in executions:
        output = execution.output
        if execution.tool_name != IMAGE_TOOL_NAME or execution.status != ToolResultStatus.SUCCESS:
            continue
        if not isinstance(output, dict):
            continue
        images = output.get("images")
        pictures: list[object] = [output]
        if isinstance(images, list):
            pictures.extend(cast(list[object], images))
        for picture in pictures:
            if isinstance(picture, dict):
                one = cast(dict[str, object], picture)
                take(prompt, one.get("prompt_added"))
                take(negative, one.get("negative_added"))
    return tuple(prompt), tuple(negative)


def _record_tools(
    speaker: Participant, turn_id: str, executions: Sequence[ToolExecutionRecord]
) -> tuple[tuple[RoomToolUse, ...], tuple[RoomWrittenFile, ...]]:
    """The room's account of a turn's tool calls, and the files they wrote (#1353, #1354).

    A path is recorded as *written* only on the tool's own declaration (#1167), only when
    the call succeeded, and only when the output names it. `file_read` returns a `path`
    as well, and the shape of an output is not a capability.

    Every other call that could have written is counted as a possible unnamed write
    (`wrote_unnamed`), so the dock can say its list may be missing files (#1366):

    * a call that reached a tool declaring `writes_files` and did not both succeed and name
      a path -- a shell that wrote and then exited non-zero or timed out, a tool that raised
      partway. Failing does not undo a write. A call that never reached the tool (unknown or
      refused) carries `writes_files=False` and is not counted;
    * any call to a tool declaring `spawns_subagents`, whatever its status. The helper runs
      with the seat's own tools and its calls never reach this turn's record, and a helper
      that failed may have written first. Its writes are counted, not propagated: the
      delegation result does not carry the helper's tool records, and reading them from
      its session would be a second, unsynchronised source.

    One call can name several files: an output's `paths` list is recorded alongside its
    `path`, each file once (#1558), in `RoomToolUse.written_paths` and as one
    `RoomWrittenFile` each. That is how a peer called through `a2a_call` reports what it
    wrote. The files come from `ToolExecutionRecord.produced_paths` (#2085): what the tool
    declared in its result's `artifacts` -- an image call's pictures, which its output names
    by link only (#2013) -- else a writing tool's `path`/`paths`.
    `RoomToolUse.written_path` stays the first of them. A writing call whose output says
    `unnamed_writes: true` -- a peer that may have written something it did not name -- is
    counted as a possible unnamed write too, as a helper is.
    """
    uses: list[RoomToolUse] = []
    written: list[RoomWrittenFile] = []
    for execution in executions:
        succeeded = execution.status is ToolResultStatus.SUCCESS
        output = execution.output
        mapping: dict[str, Any] = dict(output) if isinstance(output, dict) else {}
        # What the call produced, as the tool declared it (#2085): not its output's shape,
        # and not gated on `writes_files`, which the image tool no longer declares (#2079).
        named: list[str] = list(execution.produced_paths)
        written_path: str | None = named[0] if named else None
        subagent_id: str | None = None
        child = mapping.get("subagent_id")
        if execution.spawns_subagents and succeeded and isinstance(child, str) and child:
            subagent_id = child
        arguments, cut_arguments = _preview(unwrap_immutable(execution.arguments))
        result, cut_result = _preview(output)
        uses.append(
            RoomToolUse(
                turn_id=turn_id,
                participant_id=speaker.id,
                tool_name=execution.tool_name,
                tool_call_id=execution.tool_call_id,
                status=str(execution.status.value),
                error=execution.error,
                duration_ms=execution.duration_ms,
                arguments_preview=arguments,
                output_preview=result,
                truncated=cut_arguments or cut_result,
                written_path=written_path,
                written_paths=tuple(named),
                wrote_unnamed=(written_path is None and execution.writes_files)
                or (execution.writes_files and mapping.get("unnamed_writes") is True)
                or execution.spawns_subagents,
                subagent_id=subagent_id,
            )
        )
        written.extend(
            RoomWrittenFile(
                path=one,
                participant_id=speaker.id,
                tool_name=execution.tool_name,
                turn_id=turn_id,
                tool_call_id=execution.tool_call_id,
            )
            for one in named
        )
    return tuple(uses), tuple(written)


#: The two readings above, for a head that lands a turn without this orchestrator: a CLI,
#: ACP or A2A head's one-seat room records its turns the way a room seat's are (#1837).
record_turn_tools = _record_tools
memory_save_outcome = _memory_save_outcome
image_prompt_additions = _image_prompt_additions


class RoomOrchestrator:
    """Drives one room: appends an utterance, then runs turns until the floor is quiet."""

    def __init__(
        self,
        store: RoomStoreProtocol,
        selectors: Sequence[SpeakerSelectorProtocol],
        resolver: RoomAgentResolverProtocol,
        bus: EventBusProtocol | None = None,
        extractor: KnowledgeExtractor | None = None,
        sessions: SeatSessionSetAsideProtocol | None = None,
    ) -> None:
        self._store = store
        self._selectors = tuple(selectors)
        self._resolver = resolver
        #: What learns durable facts from each committed turn, after the room's cascade has
        #: ended (#1404). Optional like `bus`: `ucx room say` learns nothing.
        self._extractor = extractor
        #: The store the seats' own sessions are saved to, asked after each save whether it
        #: had to set an unreadable record aside (#1844). Optional: without it the record
        #: is still kept -- the store does that -- but no row says so.
        self._sessions = sessions
        #: room id -> the task working through that room's queued lessons. One per room,
        #: so a room's lessons are learned one at a time and in order.
        self._learning: dict[str, asyncio.Task[None]] = {}
        # Optional, and deliberately so. `post()` returns only once every turn has landed,
        # so without a bus a caller waits through the whole exchange with nothing to show —
        # which is what a head needs this for. A CLI that wants no streaming should not
        # have to construct a bus to say so.
        self._bus = bus
        self._publishers: dict[str, PublisherHandleProtocol] = {}
        #: room id -> (turn id, the task running it). The turn id is what makes eviction
        #: safe: keyed by room alone, a finishing turn's `finally` deleted whichever turn
        #: happened to be in the slot, which after an overlap was somebody else's live
        #: one -- so `interrupt` cancelled nothing and answered as though it had.
        self._active_turns: dict[str, tuple[str, asyncio.Task[Any]]] = {}
        #: room id -> the turn this orchestrator counted as started and has not yet landed
        #: or given up on. Set in the same synchronous step as the turn-started save and
        #: cleared only after the landing save, so a reader never sees `turns_started`
        #: ahead of `turns_landed` for a turn that is still in progress here (#1366).
        self._unlanded: dict[str, str] = {}
        self._interrupted_rooms: set[str] = set()
        #: room id -> how many human messages the room has accepted. A loop captures this
        #: when it starts and stops as soon as it differs, which is how a newer message
        #: abandons an older cascade exactly (`_interjected` reads the transcript and
        #: cannot see a cascade that has not written anything yet).
        self._generation: dict[str, int] = {}
        #: room id -> the floor. Held across `_take_turn`, so two overlapping cascades
        #: cannot both give the floor: the second waits, and then finds its generation
        #: stale and stops. "One floor, one speaker" was otherwise true only of a room
        #: nobody sent to twice.
        self._floor: dict[str, asyncio.Lock] = {}
        self._activity_events: dict[str, list[asyncio.Event]] = {}
        #: room id -> monotonic timestamp of last reported user presence
        self._presence: dict[str, float] = {}
        #: room id -> active turn metadata in flight
        self._live_turn_info: dict[str, dict[str, Any]] = {}

    # -- public surface ----------------------------------------------------------------

    def note_presence(self, room_id: str, active: bool = True) -> None:
        """Record whether a human is actively viewing this room."""
        if active:
            self._presence[room_id] = time.monotonic()
        else:
            self._presence.pop(room_id, None)

    def is_presence_active(self, room_id: str, timeout_seconds: float = 30.0) -> bool:
        """Whether user presence in this room was recorded within timeout."""
        last_ts = self._presence.get(room_id)
        if last_ts is None:
            return False
        return (time.monotonic() - last_ts) <= timeout_seconds

    async def interrupt(self, room_id: str, reason: str = "Turn was interrupted") -> None:
        """Signal an interrupt/stop to a running room.

        Cancels any currently executing agent turn for `room_id`, records the interrupted
        turn in the transcript with `completed=False` and an error notice, and terminates
        the active turn loop.
        """
        self._interrupted_rooms.add(room_id)
        self._signal_activity(room_id)
        if self._bus is not None:
            try:
                publisher = self._publishers.get("orchestrator")
                if publisher is None:
                    publisher = self._bus.register_publisher(
                        sender_id="orchestrator", source=EventSource.SYSTEM
                    )
                    self._publishers["orchestrator"] = publisher
                await publisher.publish(
                    AgentEvent(
                        type=EventType.INTERRUPT,
                        topic=f"room.{room_id}",
                        payload={"room_id": room_id, "reason": reason},
                    )
                )
            except Exception:
                logger.warning(
                    "Room %s could not announce interrupt on the bus", room_id, exc_info=True
                )

        active = self._active_turns.get(room_id)
        if active is not None and not active[1].done():
            active[1].cancel()

    def replace_selectors(self, selectors: Sequence[SpeakerSelectorProtocol]) -> None:
        """Choose speakers with `selectors` from the next turn on.

        For a chain whose routing model was replaced in Settings (#1446). A turn already
        choosing keeps the chain it started with: the loop iterates the tuple it read.
        """
        self._selectors = tuple(selectors)

    def turn_unlanded(self, room_id: str) -> bool:
        """Whether a turn this orchestrator started in `room_id` has not yet been saved.

        The one turn `RoomFileRecord.turns_started` may legitimately be ahead by. Any
        other difference is a turn whose result was lost, and a reader says so (#1366).
        """
        return room_id in self._unlanded

    def active_turn(self, room_id: str) -> dict[str, Any] | None:
        """Return in-flight turn metadata for `room_id`, if a turn is running."""
        info = self._live_turn_info.get(room_id)
        if info is not None:
            return dict(info)
        return None

    async def post(self, room_id: str, sender_id: str, content: str) -> RoomState:
        """Append an utterance and drive the resulting agent turns to a stop."""
        state = await self.accept(room_id, sender_id, content)
        return await self.resume(room_id, state.transcript[-1].seq)

    async def accept(self, room_id: str, sender_id: str, content: str) -> RoomState:
        """Record a human utterance and return, giving nobody the floor.

        The first half of `post`, split out because a head cannot use the whole of it in
        one HTTP request: `post` returns only once every turn has landed, so a route that
        awaited it would hold one request open across several model calls. Answering the
        request from here and driving `resume` behind it is what lets the surface report
        "your message is recorded, the room is thinking" instead of nothing at all.

        Every refusal `post` makes is made here, so they still reach the caller
        synchronously -- an unknown sender and an agent trying to post are answers to the
        request, not events that arrive later on a topic nobody may be reading.
        """
        state = self._require_human_sender(room_id, sender_id)
        state = self._store.save(self.append_human(state, sender_id, content))
        # Clear any prior interrupt flag when a fresh post arrives.
        self._interrupted_rooms.discard(room_id)
        self._generation[room_id] = self._generation.get(room_id, 0) + 1
        self._signal_activity(room_id)
        return state

    async def accept_command(self, room_id: str, sender_id: str, content: str) -> RoomState:
        """Record a command the person typed, such as `/loop 30s ...`, as a note (#1661).

        Shown in the conversation where it was typed, and nobody's speech: a note is kept
        out of every seat's span, the interjection check and the selector window, so no
        seat is handed `[user]: /loop 30s ...` as something to answer. For the same reason
        it does not reset the turn budget and does not stop a running exchange, which is
        what a message does. `accept`'s refusals apply: only the room's human may type a
        command. The row carries no `code`, so a head shows the command as it was typed.
        """
        state = self._require_human_sender(room_id, sender_id)
        note = RoomMessage(
            seq=len(state.transcript) + 1,
            sender_id=sender_id,
            content=content,
            kind=RoomMessageKind.NOTE,
        )
        state = self._store.save(state.model_copy(update={"transcript": (*state.transcript, note)}))
        self._signal_activity(room_id)
        return state

    def _require_human_sender(self, room_id: str, sender_id: str) -> RoomState:
        """The room, when `sender_id` is its human; refused otherwise."""
        state = self._refuse_a_head_room(self._require_room(room_id))
        sender = self._participant(state, sender_id)
        if sender is None:
            roster = ", ".join(p.id for p in state.participants)
            raise UnknownRoomParticipantError(
                f"{sender_id!r} is not a participant of room {room_id!r} "
                f"(participants: {roster or 'none'})"
            )
        if sender.kind is not ParticipantKind.HUMAN:
            # A post resets the turn budget, and that budget is G6's only structural bound
            # on an agent-to-agent run. An agent able to post could lift its own ceiling
            # indefinitely, which is the loop the ceiling exists to stop. An agent speaks
            # by being given the floor, never by posting.
            raise UnknownRoomParticipantError(
                f"{sender_id!r} is an agent of room {room_id!r}, and only a human may post; "
                f"an agent speaks when the orchestrator gives it the floor"
            )
        return state

    async def resume(self, room_id: str, baseline_seq: int) -> RoomState:
        """Drive turns for an utterance already recorded at `baseline_seq`.

        The second half of `post`. `baseline_seq` is the seq this loop answers: a human
        message beyond it, appearing while the loop runs, is the interjection signal --
        see `_interjected`. A caller that passes a stale baseline is telling the loop to
        treat the message it is answering as an interjection, which stops it immediately.
        """
        self._refuse_a_head_room(self._require_room(room_id))
        try:
            return await self._run_turns(room_id, baseline_seq, self._generation.get(room_id, 0))
        finally:
            self._start_learning(room_id)  # after the cascade, never inside it (#1404)

    async def wait_for_learning(self, room_id: str) -> None:
        """Return once the room has no lesson queued and none being learned.

        For a caller that must see what a finished exchange taught its clones -- a test, a
        command that exits. The room itself never waits for learning.
        """
        while True:
            task = self._learning.get(room_id)
            if task is None or task.done():
                if self._extractor is None or not self._extractor.has_pending(room_id):
                    return
                self._start_learning(room_id)
                continue
            await asyncio.shield(task)

    async def note_human_activity(self, room_id: str, sender_id: str = "human") -> None:
        """Record that a human is composing — a timestamp, never keystroke content.

        A hint, not a fact: it does not abandon a running loop, because a composing human
        may never send. It feeds the optional pause before a low-confidence speaker takes
        the floor; only a *sent* message ends a loop.
        """
        state = self._require_room(room_id)
        turn_state = state.turn_state.model_copy(update={"last_activity_ts": time.time()})
        self._store.save(state.model_copy(update={"turn_state": turn_state}))
        self._signal_activity(room_id)
        if self._bus is not None:
            try:
                publisher = self._publishers.get(sender_id)
                if publisher is None:
                    publisher = self._bus.register_publisher(
                        sender_id=sender_id, source=EventSource.USER
                    )
                    self._publishers[sender_id] = publisher
                await publisher.publish(
                    AgentEvent(
                        type=EventType.USER_INPUT,
                        topic=f"room.{room_id}",
                        payload={
                            "room_id": room_id,
                            "sender_id": sender_id,
                            "status": "typing",
                        },
                    )
                )
            except Exception:
                logger.warning(
                    "Room %s could not announce typing for %s on the bus",
                    room_id,
                    sender_id,
                    exc_info=True,
                )

    async def retry(self, room_id: str) -> RoomState:
        """Re-run the turn that failed, reusing the turn slot the failure already spent.

        **Why the budget is refunded rather than charged again.** The ceiling bounds
        *conversation* — how many agent utterances one human message may produce — and a
        failed turn produced none. Charging a retry would let a transient provider error
        shorten the exchange permanently, and the only remedy the user had before this
        existed (retype the question) resets the budget to zero outright, so charging was a
        penalty that the obvious workaround skipped. The automatic loop still charges a
        failed turn, and must: there, nothing else stops a persistently failing agent from
        being selected forever. Here the human is the ceiling — a retry happens only
        because somebody asked for one.

        **The chain is not re-run.** It would now see the failed row as that agent having
        already spoken and route the floor elsewhere, so the retried turn would answer as
        somebody else — which is not a retry. The floor goes back to the agent that failed,
        under the decision that gave it the floor, and that decision is recorded verbatim
        on the new message so "why did this one answer" survives.

        The failed row is **kept**. A retry appends; a room that erased its failures once
        repaired could not answer "did this have to be asked twice", which is what G8
        records them for.

        Raises:
            RoomNotFoundError: No such room.
            NothingToRetryError: The room's last utterance is not a failed turn — it
                succeeded, or somebody has spoken since.
            UnknownRoomParticipantError: The agent that failed has left the room.
        """
        state = self._refuse_a_head_room(self._require_room(room_id))
        # The last *utterance*, not the last row — see `RoomState.last_utterance`, which
        # the rendering half asks the same question of.
        failed = state.last_utterance
        if failed is None or (failed.error is None and failed.content.strip() != ""):
            spoke = failed.sender_id if failed is not None else "nobody"
            raise NothingToRetryError(
                f"The last thing said in room {room_id!r} was {spoke!r}'s, and it did not "
                f"fail. A retry re-runs a failed turn; there is none to re-run."
            )

        speaker = self._participant(state, failed.sender_id)
        if speaker is None or speaker.kind is not ParticipantKind.AGENT:
            roster = ", ".join(p.id for p in state.participants)
            raise UnknownRoomParticipantError(
                f"{failed.sender_id!r} failed a turn in room {room_id!r} and is no longer "
                f"an agent of it, so the floor cannot be given back "
                f"(participants: {roster or 'none'})"
            )

        # The refund, and it is written before the turn runs rather than netted afterwards:
        # `_take_turn` re-reads the room and increments from what it finds, so a refund held
        # in memory across a provider round trip would be lost to that re-read.
        refunded = state.turn_state.model_copy(
            update={"agent_turns_since_human": max(0, state.turn_state.agent_turns_since_human - 1)}
        )
        self._store.save(state.model_copy(update={"turn_state": refunded}))

        retried = SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=speaker.id,
            selector="orchestrator",
            reasoning=(
                f"retry of {speaker.id!r}'s "
                f"{'failed' if failed.error is not None else 'silent'} turn at seq {failed.seq}"
            ),
        )
        # Under the floor, like every other turn. `retry` is the second door into
        # `_take_turn`, and a retry running beside a live cascade put two agents on the
        # floor by the one route the loop's own guard cannot see.
        try:
            async with self._floor.setdefault(room_id, asyncio.Lock()):
                return await self._take_turn(room_id, speaker, failed.decision or retried)
        finally:
            self._start_learning(room_id)

    def append_human(self, state: RoomState, sender_id: str, content: str) -> RoomState:
        """Return `state` with a human utterance appended and the turn budget reset.

        Public because it is the seam an interjecting writer uses: a head appending a
        message mid-loop must produce exactly the state the orchestrator would have, or
        the two disagree on sequence numbering.
        """
        message = RoomMessage(seq=len(state.transcript) + 1, sender_id=sender_id, content=content)
        turn_state = state.turn_state.model_copy(
            update={
                "agent_turns_since_human": 0,
                "last_speaker_id": sender_id,
                "last_activity_ts": time.time(),
            }
        )
        return state.model_copy(
            update={
                "transcript": (*state.transcript, message),
                "turn_state": turn_state,
                "last_decision": None,
            }
        )

    # -- the turn loop -----------------------------------------------------------------

    async def _run_turns(self, room_id: str, baseline_seq: int, generation: int) -> RoomState:
        """Give the floor, one speaker at a time, until something stops the room.

        `generation` is the value `accept` had reached when this loop was started. A
        newer human message bumps it, and this loop then stops at its next check --
        which is how a second `resume` abandons the first rather than running beside it.
        """
        while True:
            if room_id in self._interrupted_rooms:
                return self._require_room(room_id)

            if self._generation.get(room_id, generation) != generation:
                # A newer human message has been accepted. That message began a loop of
                # its own, and two loops giving the floor is the one thing the floor is
                # for. Abandoned, not finished: whatever it had already recorded stands.
                return self._require_room(room_id)

            state = self._require_room(room_id)

            # (a) A human spoke while the previous turn was running. Abandon what is left
            # of this loop; their message has already begun a loop of its own.
            if self._interjected(state, baseline_seq):
                return state

            # (b) The ceiling is checked before anyone is consulted, so an exhausted
            # budget costs no selector call and no model call.
            ceiling = (
                AUTONOMOUS_CIRCUIT_BREAKER_TURNS
                if state.policy.autonomous
                else state.policy.max_agent_turns_per_human_message
            )
            if state.turn_state.agent_turns_since_human >= ceiling:
                # Recorded, like every other stop. Since one address may name more agents
                # than the ceiling allows, reaching it now routinely means somebody was
                # asked and never got the floor — and a state that says nothing reads as
                # though everything asked was answered.
                reasoning = (
                    f"the autonomous circuit breaker of {ceiling} turns was reached; "
                    f"paused to prevent infinite looping. Post a message to continue."
                    if state.policy.autonomous
                    else f"the turn ceiling of {ceiling} for one human message was reached; "
                    f"anything still outstanding was not given the floor"
                )
                reached = SpeakerDecision(
                    verdict=SelectionVerdict.SILENCE,
                    selector="orchestrator",
                    reasoning=reasoning,
                )
                return self._store.save(state.model_copy(update={"last_decision": reached}))

            if state.policy.autonomous and not self.is_presence_active(room_id):
                paused = SpeakerDecision(
                    verdict=SelectionVerdict.SILENCE,
                    selector="orchestrator",
                    reasoning="autonomous discussion paused: user is not actively viewing the room",
                )
                return self._store.save(state.model_copy(update={"last_decision": paused}))

            chain_saw = state
            decision = await self._decide(state)

            # Re-read before acting on the decision. Selection is the second-longest step
            # in the loop — an LLM link spends a provider round trip — and the roster can
            # move across it. Validating against the snapshot the chain was *given* would
            # admit a speaker who has since left, which is the opposite of what "checked
            # against the live roster at the moment the floor is given" claims. It also
            # keeps the silence write off a stale revision.
            state = self._require_room(room_id)

            # (c) The same window guard (a) covers, re-applied on what the re-read can now
            # see. Guard (a) runs *before* selection and so could never observe a message
            # that arrived during it — which is exactly the window the re-read exists for,
            # and the longest one in the loop when an LLM link is in the chain. Without
            # this, the loop saw the interjection and walked into a full agent turn
            # answering a message the human had already superseded (#710).
            if self._interjected(state, baseline_seq):
                return state

            if decision.verdict is SelectionVerdict.SILENCE:
                return self._store.save(state.model_copy(update={"last_decision": decision}))

            # (e) The named speaker was in the room when the chain answered and is not now.
            # A race, not a defect — and not a lost turn either: the floor goes back to the
            # chain against the live roster rather than ending the loop (#710). See
            # `_forgive_departure_race` for why nothing here suspends.
            speaker = self._validate_speaker(state, chain_saw, decision)
            if speaker is None:
                self._forgive_departure_race(state)  # (e)
                continue

            # (f) Hesitation pause: race a timer against human activity.
            pause = max(0.0, state.policy.hesitation_seconds * (1.0 - decision.confidence))
            if state.policy.autonomous and state.policy.hesitation_seconds > 0.0:
                pause = max(pause, AUTONOMOUS_DEFAULT_HESITATION_SECONDS)
            if pause > 0.0:
                pre_wait_activity = state.turn_state.last_activity_ts
                activity_event = asyncio.Event()
                self._activity_events.setdefault(room_id, []).append(activity_event)
                try:
                    await asyncio.wait_for(activity_event.wait(), timeout=pause)
                except TimeoutError:
                    pass
                finally:
                    events = self._activity_events.get(room_id)
                    if events is not None and activity_event in events:
                        events.remove(activity_event)
                        if not events:
                            self._activity_events.pop(room_id, None)

                if room_id in self._interrupted_rooms:  # hesitation interrupted
                    return self._require_room(room_id)

                state = self._require_room(room_id)
                if self._interjected(state, baseline_seq):
                    return state
                if state.turn_state.last_activity_ts > pre_wait_activity:
                    return state
                # The pause is a second window of up to `hesitation_seconds` in which the
                # roster can move, so the same departure is possible again and gets the
                # same answer. Answering it differently here would make the room's response
                # to a departure depend on whether hesitation happened to be switched on.
                speaker = self._validate_speaker(state, chain_saw, decision)
                if speaker is None:
                    self._forgive_departure_race(state)  # (f)
                    continue

            floor = self._floor.setdefault(room_id, asyncio.Lock())
            async with floor:
                # Re-checked under the floor: waiting for it is exactly the window in
                # which a newer message can arrive, and taking the floor after that has
                # happened is the overlap this lock exists to prevent.
                if self._generation.get(room_id, generation) != generation:
                    return self._require_room(room_id)
                if room_id in self._interrupted_rooms:
                    return self._require_room(room_id)
                state = await self._take_turn(room_id, speaker, decision)

    async def _decide(self, state: RoomState) -> SpeakerDecision:
        """Run the chain and return the first non-abstaining decision.

        A chain that abstains all the way through becomes `SILENCE` **here**, and nowhere
        else: converting "nobody had an opinion" into "nobody speaks" is a decision, and
        it is recorded as one so that a quiet room stays distinguishable from a broken
        selector — which raises rather than returning any verdict.
        """
        windowed = self.window_over_utterances(state.transcript, state.policy.transcript_window)
        request = SpeakerRequest(
            room_id=state.room_id,
            participants=state.participants,
            transcript=windowed,
            turn_state=state.turn_state,
            policy=state.policy,
        )
        consulted: list[str] = []
        for selector in self._selectors:
            decision = await selector.select(request)
            consulted.append(selector.name)
            if decision.verdict is not SelectionVerdict.ABSTAIN:
                return decision

        return SpeakerDecision(
            verdict=SelectionVerdict.SILENCE,
            selector="orchestrator",
            reasoning=(
                "the selector chain was exhausted without a judgement "
                f"(abstained: {', '.join(consulted) or 'no selectors configured'})"
            ),
        )

    async def _take_turn(
        self, room_id: str, speaker: Participant, decision: SpeakerDecision
    ) -> RoomState:
        """Hand `speaker` the floor and record what it said, or that it failed.

        The state is re-read immediately before the append rather than carried across the
        turn: an agent turn is the long window in which another writer can move the
        record, and a carried handle would be stale by exactly that much.
        """
        agent = await self._resolver.resolve(
            speaker, one_seat=is_one_seat(self._require_room(room_id).participants)
        )
        state = self._require_room(room_id)
        prompt = self._render_span(state, speaker, span_token_budget(state, agent))
        # The span this turn actually showed. The high-water mark advances to *this*, not to
        # the speaker's own new message: anything appended while the turn ran — an
        # interjection, by the very seam `append_human` exists to provide — was never
        # rendered, and marking past it would skip it permanently. That inverts the mark's
        # purpose from "do not show it twice" into "never show it at all".
        rendered_through = state.transcript[-1].seq if state.transcript else 0

        # An identity for the turn, minted before it runs. Deliberately not a row
        # number: `append_human` is a mid-turn seam, so `len(transcript) + 1` is a guess
        # that anything appended during the turn invalidates -- and the deltas were then
        # addressed to whichever row the guess collided with.
        turn_id = uuid.uuid4().hex
        # The seat's session as it stands before the turn, taken before the turn is
        # counted as started: a turn that does not commit returns the seat to exactly
        # this (#1423). Taken first so that a refusal here refuses a turn nothing counted.
        checkpoint = agent.checkpoint_turn(speaker.session_id)
        self._record_turn_started(room_id, speaker, turn_id)
        try:
            return await self._run_started_turn(
                room_id,
                speaker,
                decision,
                agent,
                prompt,
                rendered_through,
                turn_id,
                checkpoint,
            )
        finally:
            # After the landing save, or after it failed: either way this orchestrator is
            # no longer about to write the turn, so a difference between the counts that
            # remains is a lost turn and the dock says so.
            if self._unlanded.get(room_id) == turn_id:
                del self._unlanded[room_id]

    def _record_turn_started(self, room_id: str, speaker: Participant, turn_id: str) -> None:
        """Count the turn as started, in a save of its own, before it runs any tool.

        The landing save carries the row, the tools and the files together, so losing it
        -- a compare-and-swap lost to a second writer, a disk error, the process dying
        mid-turn -- used to leave no trace, and a room that was clean before read as "no
        file written" with a file on disk (#1366). This count is what survives that.

        **If this save fails the turn is refused, not run.** A turn run without it is the
        unrecorded turn the count exists to rule out. Nothing has run yet, so the refusal
        loses nothing, and it states its cause (`TurnNotStartedError`, P6). No `await`
        separates the save from `_unlanded`, so no reader sees the count ahead with no
        turn in progress.
        """
        state = self._require_room(room_id)
        record = state.file_record
        try:
            self._store.save(
                state.model_copy(
                    update={
                        "file_record": record.model_copy(
                            update={"turns_started": record.turns_started + 1}
                        )
                    }
                )
            )
        except Exception as exc:  # any failure to count the start refuses the turn
            raise TurnNotStartedError(
                f"{speaker.display_name}'s turn was not started, because the conversation "
                f"could not save that it was starting. Nothing was run, so it is safe to "
                f"try again."
            ) from exc
        self._unlanded[room_id] = turn_id

    async def _run_started_turn(
        self,
        room_id: str,
        speaker: Participant,
        decision: SpeakerDecision,
        agent: BaseAgentProtocol,
        prompt: str,
        rendered_through: int,
        turn_id: str,
        checkpoint: SessionState,
    ) -> RoomState:
        """Run a turn already counted as started, and land it with the landed count.

        **A turn commits in three places or in none (#1423):** the seat's session, the
        transcript row's answer, and the seat's `last_seen_seq`. One predicate decides all
        three -- no error, not interrupted. A turn that fails it is rolled back out of the
        seat's session before the session is written, so the span it was shown is not
        left in the conversation for the retry to stack a second copy on, and a stopped
        turn leaves no half of a tool step behind. If the room's own write fails after a
        turn that passed it, the seat is rolled back and rewritten before the failure
        propagates, so the seat does not remember a turn the room does not have.
        """
        state = self._require_room(room_id)
        await self._publish_turn_start(state, speaker, turn_id)

        error: str | None = None
        content = ""
        provenance = None
        usage = None
        completed = True
        refusal: RoomTurnRefusal | None = None
        provider_failure: ProviderFailure | None = None
        persona_edit_dropped = False
        # What the turn says its tools were. Stays empty *and unrecorded* when the turn
        # raised or was cancelled: there is no `TurnResult` to read them from, and an
        # empty list stored as recorded would claim the seat used none (P6).
        executions: Sequence[ToolExecutionRecord] = ()
        tools_recorded = False
        # A turn that raised or was cancelled has no `TurnResult`, and keeps the story
        # the room had.
        story_id = state.story_id
        turn_task = asyncio.create_task(
            agent.execute_turn(
                prompt,
                stream_callback=self._stream_callback(room_id, speaker.id, turn_id),
                caller_turn_id=turn_id,
                # Every seat gets the room's id and the room's story, so two seats in one
                # room resolve the same story and hold its lease as one writer (#1555).
                room_id=room_id,
                story_id=state.story_id,
                person_names=_person_names(state),  # per turn, never stored (#1857)
                # The room's workspace, or None for the seat's own: bound for this turn
                # only, so a change applies from the next turn (clone-data-scopes §3.6).
                workspace_root=Path(state.workspace) if state.workspace is not None else None,
            )
        )
        self._active_turns[room_id] = (turn_id, turn_task)
        try:
            result = await turn_task
        except asyncio.CancelledError:
            completed = False
            error = "Turn was interrupted"
        except Exception as exc:
            # Recorded, not raised: a room that drops a failed speaker presents as an
            # agent that chose not to answer, which is what a working quiet room also
            # looks like. The turn still spends budget, or a persistently failing agent
            # is selected forever.
            error = f"{type(exc).__name__}: {exc}"
        else:
            content = result.content
            provenance = result.provenance
            usage = result.usage
            executions = result.tool_executions
            story_id = result.story_id
            # Not simply True: a turn that failed while a step's tools were running may
            # have run calls whose records never reached the list, and storing that list
            # as the turn's account would claim they did not happen (#1366).
            tools_recorded = result.tool_executions_complete
            # `BaseAgent.execute_turn` returns a turn's failure rather than raising it, so a
            # result carrying `error` is a failed turn and lands as one. Reading only
            # `content` landed a real agent's failure as a successful empty row: nothing on
            # the landed event for a head to show over the partial deltas it was streamed
            # (#938), the unseen span consumed, and nothing for `retry` to re-run.
            # `is_completed` is not read: it means "the turn reached an answer" and is
            # False on results that never set it, while `completed` here means "not
            # interrupted".
            if result.error is not None:
                error = result.error
                # From the turn's stated stop reason, never from `error`'s wording (#969).
                refusal = turn_refusal(result.stop_reason)
                provider_failure = self._with_own_model(speaker.id, result.provider_failure)
                persona_edit_dropped = result.stop_reason == "persona_edit_failed"
        finally:
            # Only if it is still ours. Evicting somebody else's live turn made
            # `interrupt` cancel nothing while answering as though it had -- a Stop that
            # reported success against a still-generating agent. With both doors into
            # `_take_turn` holding the floor this is unreachable, and it is kept as the
            # second barrier rather than the only one: it is two lines, and the
            # alternative is that a future third door reintroduces a silent Stop. No
            # `Killed by:` declaration accompanies it, because no test can kill it while
            # the floor holds.
            current = self._active_turns.get(room_id)
            if current is not None and current[0] == turn_id:
                del self._active_turns[room_id]
            current_info = self._live_turn_info.get(room_id)
            if current_info is not None and current_info.get("turn_id") == turn_id:
                del self._live_turn_info[room_id]

        # One predicate for the whole commit (#1423): the session below, the row's answer,
        # and `last_seen_seq` all read it, so they cannot disagree about whether the
        # turn happened. An interrupted turn is covered by `error` too: the cancel branch
        # above sets it, so `completed` adds nothing here. A silent turn (empty content
        # with no tool executions) did not answer, so its unseen span is not consumed (#2168).
        committed = error is None and (bool(content.strip()) or bool(executions))
        rollback_error = None if committed else self._roll_back_seat(agent, speaker, checkpoint)

        # The seat's own session -- the model context, tool calls and results this
        # transcript deliberately leaves out -- is written here, after every turn however
        # it ended, because nothing else writes it. `resolve` hydrates it before a seat's
        # first turn, and that resume read an empty store: `sessions/core/` held no seat
        # record after any number of room turns, so a restarted room's agents answered
        # from a blank history under a transcript that said otherwise. A failed or
        # interrupted turn is written too, as the chat route writes one: the agent in
        # memory is what the next turn uses, and a restart must not see a different one.
        # Written *after* the rollback, so what a restart sees is the seat without the
        # failed turn, and the log still holds the turn's events and the rollback's.
        # A rollback that could not be made is not written over, and the row says so.
        persist_error = rollback_error or self._persist_seat(agent, speaker)

        # Re-read, with no `await` from here to the save: a change this server's event loop
        # made while the turn ran -- a rename, a story released from the Files screen -- is
        # carried into the landing rather than refused as a stale write that loses the reply
        # (#1578). The state read when the turn started is the wrong base for this save.
        # This holds within one process only: `RoomStore.save` checks the revision and then
        # replaces the file without a lock, so a second server on the same store can still
        # race this save.
        state = self._require_room(room_id)  # the landing's own read
        # A human has spoken beyond the span this turn saw: the selection that gave
        # `speaker` the floor answered a message that has since been superseded, so
        # `decision` is a stale judgement of it -- the same rule `_interjected` already
        # applies to the loop (§3.8), applied here to the write this turn is about to
        # make. The message the human sent instead already has its own, later decision
        # on `state.last_decision`, or will get one; this turn's must not clobber it (#945).
        superseded = self._interjected(state, rendered_through)
        memory_facts_tried, memory_facts_unsaved = _memory_save_outcome(executions)
        image_prompt_added, image_negative_added = _image_prompt_additions(executions)
        message = RoomMessage(
            seq=len(state.transcript) + 1,
            sender_id=speaker.id,
            # The seat's session, recorded with the row: the trace is read from it, even
            # after the seat is re-keyed (clone-data-scopes §4 step 3).
            session_id=speaker.session_id or None,
            content=content,
            decision=decision,
            provenance=provenance,
            error=error,
            refusal=refusal,
            provider_failure=provider_failure,
            persona_edit_dropped=persona_edit_dropped,
            completed=completed,
            rendered_through=rendered_through,
            persist_error=persist_error,
            session_set_aside=self._session_set_aside(speaker),
            memory_facts_tried=memory_facts_tried,
            memory_facts_unsaved=memory_facts_unsaved,
            image_prompt_added=image_prompt_added,
            image_negative_added=image_negative_added,
            turn_id=turn_id,
            tools_recorded=tools_recorded,
            usage=usage,
        )
        uses, written = _record_tools(speaker, turn_id, executions)
        turn_state = state.turn_state.model_copy(
            update={
                "agent_turns_since_human": state.turn_state.agent_turns_since_human + 1,
                "last_speaker_id": speaker.id,
            }
        )
        seen = dict(state.last_seen_seq)
        if committed:
            # Advanced only on success, so a failed turn is retried against the same
            # unseen span instead of silently consuming it.
            seen[speaker.id] = str(rendered_through)

        landing = state.model_copy(
            update={
                "transcript": (*state.transcript, message),
                "turn_state": turn_state,
                "last_seen_seq": seen,
                "last_decision": state.last_decision if superseded else decision,
                # In the same write as the row, so the ledger and the transcript
                # cannot disagree about whether this turn happened (P8).
                "tool_uses": (*state.tool_uses, *uses),
                "written_files": (*state.written_files, *written),
                # The story a `story_library` call in this turn opened, created or closed
                # (#1555), as the seat's lifecycle hooks moved it (#1775) -- the room does
                # not re-derive it from the records. Kept whether or not the turn then
                # failed: the lease moved when the call succeeded, and a room that forgot
                # it would hold a lease it does not know it has.
                "story_id": story_id,
                # What `written_files` cannot see, counted where it happens and kept
                # past any clear or rewind that removes the rows above (#1366).
                "file_record": state.file_record.model_copy(
                    update={
                        "unrecorded_turns": state.file_record.unrecorded_turns
                        + (0 if tools_recorded else 1),
                        "unattributed_writes": state.file_record.unattributed_writes
                        + sum(1 for u in uses if u.wrote_unnamed),
                        # In the same write as everything the turn produced, so it
                        # moves only if they all reached the store (#1366).
                        "turns_landed": state.file_record.turns_landed + 1,
                    }
                ),
            }
        )
        try:
            saved = self._store.save(landing)
        except Exception as exc:
            # The room did not take the turn, so the seat must not keep it: a seat whose
            # session holds a reply the transcript does not would answer the next span
            # from a turn nobody in the room saw. Rolled back and rewritten, then the
            # room's failure propagates.
            if committed:
                self._undo_committed_seat(agent, speaker, checkpoint)
            if isinstance(exc, RoomError):
                # Already a refusal written for a person (a lost compare-and-swap).
                raise
            # A fault's own text -- an `OSError` names the store's path -- is for the
            # log. What reaches the conversation says what happened in plain words, and
            # the cause stays chained for anyone who catches it (P6, #1495).
            logger.error(
                "Could not save the turn of room seat %r in room %s; the seat was put "
                "back to before the turn",
                speaker.id,
                room_id,
                exc_info=exc,
            )
            raise TurnNotLandedError(
                f"{speaker.display_name}'s reply could not be saved to this conversation, "
                "so it was not kept. The reason is in the server log."
            ) from exc
        if committed:  # a failed turn teaches nothing
            # Queued, not run: learning waits for the cascade to end (#1404).
            self._queue_lesson(
                state, speaker, agent, turn_id, rendered_through, content, executions
            )
        # Before the landed reply, so a head sees the calls ahead of the answer they fed.
        await self._publish_tools(saved, speaker, message.seq, uses)
        await self._publish(saved, message, turn_id)
        return saved

    # -- learning from a turn (#1404) ----------------------------------------------------

    def _queue_lesson(
        self,
        state: RoomState,
        speaker: Participant,
        agent: BaseAgentProtocol,
        turn_id: str,
        rendered_through: int,
        content: str,
        executions: Sequence[ToolExecutionRecord],
    ) -> None:
        """Queue what `speaker` saw and did in a committed turn, for learning later.

        `state` is the room as the landing read it, before the landing: its
        `last_seen_seq` for the speaker still marks where this turn's span began.
        """
        if self._extractor is None or not isinstance(agent, LearningSeatProtocol):
            return
        memory = agent.memory
        if memory is None:
            return
        try:
            since = int(state.last_seen_seq.get(speaker.id, "0"))
        except ValueError:
            since = 0
        lines = _lesson_lines(state, speaker, since, rendered_through, content, executions)
        self._extractor.submit(
            Lesson(
                clone_id=speaker.id,
                clone_name=speaker.display_name,
                room_id=state.room_id,
                session_id=speaker.session_id or state.room_id,
                turn_id=turn_id,
                lines=lines,
                memory=memory,
                generate=agent.invoke_auxiliary_model,
                person_names=_person_names(state),
                clone_names=_clone_names(state, speaker),
            )
        )

    def _start_learning(self, room_id: str) -> None:
        """Start working through the room's queued lessons, unless that is under way."""
        if self._extractor is None or not self._extractor.has_pending(room_id):
            return
        running = self._learning.get(room_id)
        if running is not None and not running.done():
            return  # it takes the new lesson when it gets to it
        self._learning[room_id] = asyncio.create_task(self._learn(room_id))

    async def _learn(self, room_id: str) -> None:
        """Learn each queued lesson in turn, each once no turn holds the floor.

        The floor is waited for and let go, not held: learning takes a model call, and a
        reply the person is waiting for must not queue behind it.
        """
        extractor = self._extractor
        if extractor is None:
            return
        while True:
            waited = self._floor.setdefault(room_id, asyncio.Lock())
            async with waited:  # wait for the floor, then let it go
                pass
            lesson = extractor.take(room_id)
            if lesson is None:
                return
            try:
                outcome = await extractor.extract(lesson)
            except asyncio.CancelledError:
                raise
            except Exception:  # extract returns its failures; this is a defect in it
                logger.exception("Learning from turn %s in room %s failed", lesson.turn_id, room_id)
                outcome = ExtractionOutcome(
                    clone_id=lesson.clone_id,
                    room_id=lesson.room_id,
                    turn_id=lesson.turn_id,
                    error=EXTRACTION_FAILED,
                )
            if outcome.added or outcome.error is not None:
                self._record_learning(outcome)
            await self._publish_learning(outcome)

    def _record_learning(self, outcome: ExtractionOutcome) -> None:
        """Write what the turn taught onto its row: the saved fact ids, or the failure.

        Re-read with no `await` before the save, as the landing is. A row that is gone -- the
        history was cleared or rewound meanwhile -- has nothing to carry it, and the facts
        stay saved either way.
        """
        try:
            state = self._require_room(outcome.room_id)
            rows = list(state.transcript)
            for index, row in enumerate(rows):
                if row.turn_id == outcome.turn_id and row.sender_id == outcome.clone_id:
                    rows[index] = row.model_copy(
                        update={
                            "knowledge_learned": (*row.knowledge_learned, *outcome.added),
                            "knowledge_extract_error": outcome.error,
                        }
                    )
                    break
            else:
                return
            self._store.save(state.model_copy(update={"transcript": tuple(rows)}))
        except Exception:
            logger.warning(
                "Room %s could not note on turn %s what %s learned from it",
                outcome.room_id,
                outcome.turn_id,
                outcome.clone_id,
                exc_info=True,
            )

    async def _publish_learning(self, outcome: ExtractionOutcome) -> None:
        """Say on the room's knowledge topic that a clone's memory may have changed.

        A sub-topic, like tool calls, so a head's reducer for the room's own events never
        sees it; the head refreshes the room and its memory views.
        """
        if self._bus is None:
            return
        try:
            await self._publisher(outcome.clone_id).publish(
                AgentEvent(
                    type=EventType.KNOWLEDGE_UPDATED,
                    topic=f"room.{outcome.room_id}.knowledge",
                    payload={
                        "room_id": outcome.room_id,
                        "agent_id": outcome.clone_id,
                        "turn_id": outcome.turn_id,
                        "added": len(outcome.added),
                        "reinforced": len(outcome.reinforced),
                        "superseded": len(outcome.superseded),
                        "failed": outcome.error is not None,
                    },
                )
            )
        except Exception:
            logger.warning(
                "Room %s could not announce what %s learned on the bus",
                outcome.room_id,
                outcome.clone_id,
                exc_info=True,
            )

    async def _publish_tools(
        self, state: RoomState, speaker: Participant, seq: int, uses: Sequence[RoomToolUse]
    ) -> None:
        """Announce a landed turn's tool calls on `room.{room_id}.tool` (#1353).

        **When the turn lands, not live.** `BaseAgent` reports a turn's tools on its
        `TurnResult`, and forwarding its per-step stream would put a seat's in-flight
        scratchpad on a channel every head reads. So each call is published once, from
        the room's own record, after the write -- a subscriber that reacts by reading the
        room's history finds the call it was told about.

        **Its own topic.** The head folds exactly `room.{room_id}` into the transcript and
        treats an unknown event type there as a fault; a `.tool` sub-topic keeps these out
        of that reducer while `/api/stream`, which subscribes to everything, still carries
        them. Guarded like `_publish`: losing an announcement degrades a head, and must not
        cost the room its turn.
        """
        if self._bus is None or not uses:
            return
        try:
            publisher = self._publisher(speaker.id)
            topic = f"room.{state.room_id}.tool"
            for use in uses:
                base: dict[str, Any] = {
                    "room_id": state.room_id,
                    "participant_id": speaker.id,
                    "session_id": speaker.session_id,
                    "turn_id": use.turn_id,
                    "seq": seq,
                    "tool_call_id": use.tool_call_id,
                    "name": use.tool_name,
                }
                await publisher.publish(
                    AgentEvent(
                        type=EventType.TOOL_CALL,
                        topic=topic,
                        payload={**base, "arguments_preview": use.arguments_preview},
                    )
                )
                await publisher.publish(
                    AgentEvent(
                        type=EventType.TOOL_RESULT,
                        topic=topic,
                        payload={
                            **base,
                            "status": use.status,
                            "error": use.error,
                            "duration_ms": use.duration_ms,
                            "output_preview": use.output_preview,
                            "truncated": use.truncated,
                            "written_path": use.written_path,
                            "subagent_id": use.subagent_id,
                        },
                    )
                )
        except Exception:
            logger.warning(
                "Room %s could not announce %s's tool calls (seq %s) on the bus; they are "
                "recorded on the room and the loop continues",
                state.room_id,
                speaker.id,
                seq,
                exc_info=True,
            )

    def _publisher(self, sender_id: str) -> PublisherHandleProtocol:
        """This speaker's one publishing handle, registered on first use.

        Registered once per speaker, not once per turn. The bus keys its registry by sender
        id, so re-registering replaces that agent's handle with a fresh unrestricted one on
        every turn — no effect while nothing reads the registry back, and an authorization
        hole the moment something does.
        """
        assert self._bus is not None
        publisher = self._publishers.get(sender_id)
        if publisher is None:
            publisher = self._bus.register_publisher(sender_id=sender_id, source=EventSource.AGENT)
            self._publishers[sender_id] = publisher
        return publisher

    async def _publish(self, state: RoomState, message: RoomMessage, turn_id: str) -> None:
        """Announce one landed utterance on `room.{room_id}`, if anyone is listening.

        Published *after* the write, so a subscriber that reacts by reading the room sees
        the message it was told about. A failed turn is announced too, carrying its
        `error`: a head that only hears about successes shows a room that went quiet for no
        stated reason, which is the outcome recording the failure exists to prevent.

        `provenance` is passed through exactly as the turn produced it, `None` included.
        The room does not manufacture attribution for a value it did not produce, and does
        not vouch for an agent that omitted its own.
        """
        if self._bus is None:
            return
        payload: dict[str, Any] = {
            "room_id": state.room_id,
            "agent_id": message.sender_id,
            "seq": message.seq,
            # Named rather than left implicit. The topic now carries three kinds of
            # AGENT_REPLY -- the floor being taken, a token delta, and this -- and a
            # consumer that selected the landed one by the *absence* of a status silently
            # started matching every delta the moment streaming was added.
            "status": "final",
            # The turn the deltas named. A head accumulates a live bubble against this
            # and clears it here; matching on the row number could not work, because the
            # deltas were published before the row was known.
            "turn_id": turn_id,
            "content": message.content,
            "completed": message.completed,
        }
        if message.error is not None:
            payload["error"] = message.error
        if message.refusal is not None:
            payload["refusal"] = message.refusal.value
        if message.provider_failure is not None:
            payload["provider_failure"] = message.provider_failure.model_dump(mode="json")
        try:
            publisher = self._publishers.get(message.sender_id)
            if publisher is None:
                # Registered once per speaker, not once per turn. The bus keys its registry
                # by sender id, so re-registering replaces that agent's handle with a fresh
                # unrestricted one on every turn — no effect while nothing reads the
                # registry back, and an authorization hole the moment something does.
                publisher = self._bus.register_publisher(
                    sender_id=message.sender_id, source=EventSource.AGENT
                )
                self._publishers[message.sender_id] = publisher
            await publisher.publish(
                AgentEvent(
                    type=EventType.AGENT_REPLY,
                    topic=f"room.{state.room_id}",
                    payload=payload,
                    provenance=message.provenance,
                )
            )
        except Exception:
            # **A presentation channel must not be able to abort a conversation.** The bus
            # raises on ordinary operating conditions — stopped, queue full under an ERROR
            # backpressure policy, an unauthorized topic — and under BLOCK it waits on a
            # slow subscriber. Awaiting that unguarded inside the turn loop let a head's
            # problem cost the room its remaining turns and hand the caller an exception
            # instead of a `RoomState`. The turn is already written; losing its
            # announcement is a degraded head, while losing the turn is a lost conversation.
            logger.warning(
                "Room %s could not announce %s's turn (seq %s) on the bus; the turn is "
                "recorded and the loop continues",
                state.room_id,
                message.sender_id,
                message.seq,
                exc_info=True,
            )

    def _with_own_model(
        self, participant_id: str, failure: ProviderFailure | None
    ) -> ProviderFailure | None:
        """``failure``, naming the clone and its own model when that model is the cause (§3.6).

        A clone that names its own model ref and finds it gone, unreachable or unlisted is
        not moved to the system default: the row says which clone and which model, and
        offers the one action ``use_system_default`` (model-gateway §3.6, G7). The turn is
        not re-run.
        """
        if failure is None or failure.kind not in _OWN_MODEL_FAILURES:
            return failure
        pinned = getattr(self._resolver, "pinned_model", None)
        found: object = pinned(participant_id) if callable(pinned) else None
        if not isinstance(found, tuple):
            return failure
        parts = cast(tuple[object, ...], found)
        if len(parts) != 2 or not all(isinstance(part, str) for part in parts):
            return failure
        name, ref = str(parts[0]), str(parts[1])
        return failure.model_copy(
            update={
                "message": f"{name} is set to use its own model, {ref}, and it could not be "
                f"used. {failure.message}",
                "clone": name,
                "model_ref": ref,
                "action": "use_system_default",
            }
        )

    def _stream_callback(
        self, room_id: str, agent_id: str, turn_id: str
    ) -> Callable[[str, dict[str, Any]], Awaitable[None]] | None:
        """Forward an agent's token deltas onto `room.{room_id}` while its turn runs.

        `None` when there is no bus, and that is the whole of the opt-out: a room with no
        listener hands the agent no callback, so it pays nothing per token and `G7` is
        untouched. A CLI still constructs no bus to say so.

        **`token` and `status` updates are forwarded.** An agent's thinking deltas and raw
        tool traces are its own session's (G1, G2); conversational tokens and lightweight
        progress notices (e.g. running a tool) reach the room's channel to inform the UI.
        A turn's tool calls do reach the bus, but not from here: `_publish_tools` announces
        them once the turn has landed, from the room's bounded record of them and on a
        topic of their own (#1353).

        Scoped, because the stronger reading is false: `execute_turn` invokes its callback
        once per *step* of the reasoning loop, while `TurnResult.content` is the last
        step's text. So on a tool-using turn the deltas include prose the transcript will
        never hold, and it disappears when the row lands. What this guarantees is that only
        `token` and `status` events are forwarded -- not that everything forwarded survives
        into the transcript.

        **The first publishing failure ends streaming for this turn.** `_publish` already
        tolerates a failing bus once per utterance; doing the same per chunk turns one
        broken subscriber into an exception per token in the middle of a turn. A dropped
        stream degrades to the landed reply, which is published separately and still
        arrives -- so the failure costs the head its live view and never the conversation.
        """
        if self._bus is None:
            return None

        live = True

        async def forward(event_name: str, data: dict[str, Any]) -> None:
            nonlocal live
            if not live:
                return

            if event_name == "status" and data.get("status") == "calling_tool":
                detail = data.get("detail")
                if not detail:
                    return
                info = self._live_turn_info.get(room_id)
                if info is not None and info.get("turn_id") == turn_id:
                    info["status"] = "status_update"
                    info["detail"] = str(detail)
                try:
                    publisher = self._publishers.get(agent_id)
                    if publisher is None:
                        assert self._bus is not None
                        publisher = self._bus.register_publisher(
                            sender_id=agent_id, source=EventSource.AGENT
                        )
                        self._publishers[agent_id] = publisher
                    await publisher.publish(
                        AgentEvent(
                            type=EventType.AGENT_REPLY,
                            topic=f"room.{room_id}",
                            payload={
                                "room_id": room_id,
                                "agent_id": agent_id,
                                "turn_id": turn_id,
                                "status": "status_update",
                                "detail": str(detail),
                            },
                        )
                    )
                except Exception:
                    live = False
                    logger.warning(
                        "Room %s could not stream %s's status update (%s); the turn continues",
                        room_id,
                        agent_id,
                        turn_id,
                        exc_info=True,
                    )
                return

            if event_name != "token":
                return
            delta = data.get("content")
            if not delta:
                return
            info = self._live_turn_info.get(room_id)
            if info is not None and info.get("turn_id") == turn_id:
                info["status"] = "streaming"
                info["accumulated_text"] = info.get("accumulated_text", "") + delta
            try:
                publisher = self._publishers.get(agent_id)
                if publisher is None:
                    assert self._bus is not None
                    publisher = self._bus.register_publisher(
                        sender_id=agent_id, source=EventSource.AGENT
                    )
                    self._publishers[agent_id] = publisher
                await publisher.publish(
                    AgentEvent(
                        type=EventType.AGENT_REPLY,
                        topic=f"room.{room_id}",
                        payload={
                            "room_id": room_id,
                            "agent_id": agent_id,
                            "turn_id": turn_id,
                            "status": "streaming",
                            "delta": delta,
                        },
                    )
                )
            except Exception:
                live = False
                logger.warning(
                    "Room %s could not stream %s's turn (%s); the turn continues and "
                    "its landed reply is published as usual",
                    room_id,
                    agent_id,
                    turn_id,
                    exc_info=True,
                )

        return forward

    async def _publish_turn_start(
        self, state: RoomState, speaker: Participant, turn_id: str
    ) -> None:
        """Announce that a selected agent has begun generating.

        Published on `room.{room_id}` with `status="generating"` so listening heads
        learn that an agent has taken the floor before `execute_turn` awaits.

        Carries `turn_id` and no `seq`. The row this turn lands on is not known yet --
        see `_take_turn` -- and a number that is wrong is worse than one that is absent.
        """
        payload: dict[str, Any] = {
            "room_id": state.room_id,
            "agent_id": speaker.id,
            "turn_id": turn_id,
            "status": "generating",
        }
        self._live_turn_info[state.room_id] = {
            "in_flight": True,
            "agent_id": speaker.id,
            "turn_id": turn_id,
            "status": payload["status"],
            "detail": None,
            "accumulated_text": "",
        }
        if self._bus is None:
            return
        try:
            publisher = self._publishers.get(speaker.id)
            if publisher is None:
                publisher = self._bus.register_publisher(
                    sender_id=speaker.id, source=EventSource.AGENT
                )
                self._publishers[speaker.id] = publisher
            await publisher.publish(
                AgentEvent(
                    type=EventType.AGENT_REPLY,
                    topic=f"room.{state.room_id}",
                    payload=payload,
                )
            )
        except Exception:
            logger.warning(
                "Room %s could not announce turn start for %s (%s) on the bus; the loop continues",
                state.room_id,
                speaker.id,
                turn_id,
                exc_info=True,
            )

    def _signal_activity(self, room_id: str) -> None:
        """Signal any waiting hesitation timer that activity or an interjection occurred."""
        events = self._activity_events.get(room_id)
        if events:
            for ev in tuple(events):
                ev.set()

    # -- helpers -----------------------------------------------------------------------

    def _require_room(self, room_id: str) -> RoomState:
        state = self._store.load(room_id)
        if state is None:
            raise RoomNotFoundError(f"No room {room_id!r} in the store")
        return state

    @staticmethod
    def _refuse_a_head_room(state: RoomState) -> RoomState:
        """`state`, unless a head keeps it (#1885): then no turn is driven here.

        A head's room has one writer, the head, which runs the clone on the seat's session.
        A post, a retry or a resumed cascade from here would run that session a second time
        beside it. Checked on the record as loaded, so a head room another surface has
        open is refused as soon as it asks.
        """
        head = state.head
        if head is not None:
            raise HeadRoomWriteError(head_room_write_refusal(head))
        return state

    @staticmethod
    def _persist_seat(agent: BaseAgentProtocol, speaker: Participant) -> str | None:
        """Write `speaker`'s own session, returning why it could not be written, or `None`.

        Recorded rather than raised, and not as the turn's `error`: the reply is real and
        lands either way. `error` is what `retry` and `last_seen_seq` read as "this turn
        failed", and re-running a good answer because its bookkeeping write failed would
        spend a turn to produce a second, different reply. What the caller must not do is
        land the row saying nothing -- the reply would then look durable and vanish from
        the seat's context on the next restart (P6).
        """
        try:
            agent.persist_session(speaker.session_id)
        except Exception as exc:
            logger.warning(
                "Could not write the session %s of room seat %r after its turn; the reply "
                "landed but will not be in that seat's context after a restart: %s",
                speaker.session_id,
                speaker.id,
                exc,
                exc_info=True,
            )
            return f"{type(exc).__name__}: {exc}"
        return None

    @staticmethod
    def _roll_back_seat(
        agent: BaseAgentProtocol, speaker: Participant, checkpoint: SessionState
    ) -> str | None:
        """Undo a turn that did not commit out of `speaker`'s session (#1423).

        Returns `None` when the seat is back where it was before the turn, or the text
        the row carries when it could not be put back. In that case the session is left
        unwritten, and the row says so rather than presenting the seat as clean (P6). Plain words only: the row is read by a person, and the cause is
        in the log.
        """
        try:
            agent.roll_back_turn(checkpoint, reason="turn_not_committed")
        except Exception:
            logger.exception(
                "Could not roll back the failed turn of room seat %r (session %s); its "
                "session was not written, and its next turn may still see the failed one",
                speaker.id,
                speaker.session_id,
            )
            return (
                f"{speaker.display_name}'s memory of this conversation could not be put "
                "back to where it was before this turn, so it was not saved. Its next turn "
                "may still see what this one was shown."
            )
        return None

    @classmethod
    def _undo_committed_seat(
        cls, agent: BaseAgentProtocol, speaker: Participant, checkpoint: SessionState
    ) -> None:
        """Take a turn the room could not save back out of `speaker`'s session (#1423).

        Called only on the way to re-raising the room's own failure, so nothing here may
        replace it: a failure to undo is logged beside it, and the room's error is the
        one the caller sees.
        """
        if cls._roll_back_seat(agent, speaker, checkpoint) is not None:
            return
        # The session was written with the turn in it moments ago; this rewrites it
        # without. A failure here is logged by `_persist_seat` itself.
        cls._persist_seat(agent, speaker)

    def _session_set_aside(self, speaker: Participant) -> bool:
        """Whether `speaker`'s unreadable session record was set aside when saved (#1844).

        Taken, not peeked: the store answers `True` once, so the notice is on this row and
        on no later one.
        """
        if self._sessions is None:
            return False
        return self._sessions.take_set_aside(speaker.session_id)

    @staticmethod
    def _participant(state: RoomState, participant_id: str) -> Participant | None:
        return next((p for p in state.participants if p.id == participant_id), None)

    @staticmethod
    def _interjected(state: RoomState, baseline_seq: int) -> bool:
        """True when a human has *spoken* since the message this loop answers.

        Read off the row, not off the roster. Every agent utterance carries the
        `SpeakerDecision` that gave it the floor, and only a human can `post`, so a row
        nobody was selected for is a human's. Deriving the human set from the *live* roster
        instead made a poster who left during selection invisible — their own interjection
        stopped counting the moment they were unseated, and the loop carried on answering
        the message they had superseded.

        `is_utterance`, not merely an absent decision: a membership row carries no decision
        either, since nobody selects a join, so without the guard the room's own prose
        about the roster would read as somebody cutting in and abandon the remaining turns
        for a message nobody sent.
        """
        return any(
            m.seq > baseline_seq and m.is_utterance and m.decision is None for m in state.transcript
        )

    @staticmethod
    def window_over_utterances(
        transcript: tuple[RoomMessage, ...], window: int
    ) -> tuple[RoomMessage, ...]:
        """Return the trailing transcript slice containing at most `window` utterances.

        `RoomPolicy.transcript_window` bounds what a selector reads as conversation, and
        `RoomPolicy._window_must_outlast_the_turn_ceiling` validates that the window
        exceeds `max_agent_turns_per_human_message` so that agent turns generated for an
        address do not push the human message out of the selector's view before all
        addressed agents get the floor.

        Counting raw rows in that window broke that invariant when membership rows (JOIN /
        LEAVE) were introduced into the transcript: one JOIN row inside the window allowed
        turns to push the addressing human utterance out early, dropping outstanding
        addresses silently. Windowing back to the N-th trailing utterance preserves any
        interleaved membership context while ensuring that exactly `window` utterances are
        held within the selector's view.
        """
        if window <= 0:
            return ()
        utterances_seen = 0
        for idx in range(len(transcript) - 1, -1, -1):
            if transcript[idx].is_utterance:
                utterances_seen += 1
                if utterances_seen == window:
                    return transcript[idx:]
        return transcript

    def _forgive_departure_race(self, state: RoomState) -> None:
        """Count one survived departure race, then let the caller re-consult the chain.

        **Why it is counted at all.** While a departure ended the loop, the race left a
        `SILENCE` row naming who had left. Re-selecting removes that row — a recovered race
        produces no utterance, no decision and no error — so this count is the only trace
        the race ever happened, and a room that re-selected ten times would otherwise be
        indistinguishable from one that never raced (P6, #755). The measurement is
        unchanged by the recovery: it has always counted *departures forgiven*, not turns
        lost.

        **Why this must not suspend, and why that is a property of the whole path.** The
        caller takes no turn on a fall-through, so the step ceiling at (b) cannot bound the
        re-selection. What bounds it is the snapshot: the next iteration re-reads the
        roster the chain will be given, which no longer holds the departed agent, so naming
        it a second time is a defect by construction and `_validate_speaker` raises. That
        argument holds only while nothing can put the agent back between the `continue` and
        that re-read — and what guarantees it is not a lock but the plain absence of an
        `await` from here to the top of the loop. A backoff `sleep` added anywhere on that
        path would remove the bound silently;
        `test_the_fall_through_reaches_the_next_snapshot_without_suspending` is what
        notices, which is why it runs an adversary that re-admits the departed agent at
        every opportunity the loop gives it.
        """
        turn_state = state.turn_state.model_copy(
            update={"races_forgiven": state.turn_state.races_forgiven + 1}
        )
        self._store.save(state.model_copy(update={"turn_state": turn_state}))

    def _validate_speaker(
        self, state: RoomState, chain_saw: RoomState, decision: SpeakerDecision
    ) -> Participant | None:
        """Hold a named speaker to the live roster, distinguishing a race from a defect.

        Checked here and not in the selector: a selector naming a participant who has since
        left is an ordinary event in a room whose roster changes, and the selector has no
        way to know the roster moved after it answered.

        **Two unlike cases wore one exception until #710.** A speaker the roster *had* when
        the chain was asked and no longer has is a race, and the code can identify it
        precisely by comparing the live roster against the one the chain was given; `None`
        comes back and the caller gives the floor back to the chain. A speaker the roster
        **never** had is a selector defect — no roster change explains it — and keeps the
        raise, because a wrong selector must be loud. Before the split, an ordinary
        departure reached the caller of `post()` as an error, and a room could be made to
        fail by a participant leaving at the wrong moment.

        The raise is also what bounds the caller's re-selection: once the departure has
        landed, `chain_saw` no longer holds the departed agent, so a selector stuck on it
        reaches this branch on the second round. See `_forgive_departure_race`.

        Returns:
            The speaker, or `None` when the chain was shown it and it has since left.
        """
        # `chain_saw` first, and that order is the whole point. Asking the live roster
        # first short-circuits the defect check, so a selector that named someone it was
        # never shown is rewarded whenever the name happens to have become valid — a
        # participant who joined during selection, or one who was a human when the chain
        # was asked and is an agent now. The selector sees exactly `chain_saw.participants`,
        # so a name outside it is a defect by construction, whatever the roster did next.
        seen_by_chain = (
            self._participant(chain_saw, decision.speaker_id) if decision.speaker_id else None
        )
        if seen_by_chain is None or seen_by_chain.kind is not ParticipantKind.AGENT:
            offered = ", ".join(
                p.id for p in chain_saw.participants if p.kind is ParticipantKind.AGENT
            )
            raise SpeakerSelectionError(
                f"Selector {decision.selector!r} named {decision.speaker_id!r} to speak in "
                f"room {state.room_id!r}, which was not an agent of the room when the chain "
                f"was consulted (it was offered: {offered or 'none'})"
            )

        # `seen_by_chain` proves `speaker_id` is a non-None agent id, so the live lookup
        # below is total.
        speaker = self._participant(state, seen_by_chain.id)
        if speaker is None or speaker.kind is not ParticipantKind.AGENT:
            return None
        return speaker

    @staticmethod
    def _render_span(state: RoomState, speaker: Participant, budget: int) -> str:
        """Render the transcript span `speaker` has not been shown, speakers named.

        Everything earlier already sits in that agent's own session, so re-sending it
        would duplicate the conversation inside the session rather than inform it. Two
        kinds of message are skipped: the speaker's own, which are already in its session
        as its own assistant turns and would be a duplicate and a misattribution at once;
        and those recording a failed turn, which carry no content, since an empty
        utterance attributed to an agent is not something another agent can read. Rows
        that are not utterances are skipped for a different reason: a membership row is
        the *room's* prose about who joined or left, and rendering it as
        `[critic]: critic joined the room` would hand the speaker a sentence critic never
        said; a note (`RoomMessageKind.NOTE`, the `/loop` help and status) is written for
        the people reading the room and is not seat context (design doc
        `llm-request-layering.md` §8 Q4). An agent learns the roster from the roster, not
        from a line it would read as a remark.

        **Bounded by `budget` estimated tokens, and the bound is announced** (#1641). See
        `span_token_budget` for where the figure comes from. The newest lines are kept,
        as a contiguous run back from the latest, so the span has no hole in its middle.
        When lines are dropped the prompt says how many: the agent has never seen them, so
        a silent trim would leave it answering from a gap it cannot know is there. That
        notice is paid for out of the same budget (#1661).

        The latest line is always present, because it is the message the turn answers,
        but not always whole: a line over the budget on its own is cut to fit, with its
        sender kept and the cut announced at its end (#1661). Kept whole, one pasted log
        could still fill the seat's window. The one exception is a budget smaller than
        the sender and the announcement themselves, which `span_token_budget` can give a
        seat with a window under `SPAN_WINDOW_SHARE` tokens: those two are still sent.
        """
        last_seen = int(state.last_seen_seq.get(speaker.id, "0"))
        # Rows that are not utterances and the speaker's own turns are not conversation
        # *for this speaker*: the first is the room narrating itself, the second is already
        # in its session as its own assistant turn. Both are excluded from the span and from
        # the count of what was withheld -- the notice speaks about the conversation it
        # dropped.
        owed = [
            m
            for m in state.transcript
            if m.seq > last_seen and m.is_utterance and m.sender_id != speaker.id
        ]
        said = [(f"[{m.sender_id}]: ", m.content) for m in owed if m.content]
        lines = [prefix + content for prefix, content in said]

        def run_back(allowance: int) -> int:
            """How many lines, back from the latest, fit in `allowance`; at least one."""
            kept = 0
            spent = 0
            for line in reversed(lines):
                # One token for the newline that joins it to the next line.
                cost = estimate_text_tokens(line) + 1
                if kept and spent + cost > allowance:
                    break
                spent += cost
                kept += 1
            return kept

        allowance = budget
        kept = run_back(allowance)
        if kept < len(owed):
            # Lines are withheld, so the notice is sent, and it is paid for first. Costed
            # at the count of every owed line, which is as many digits as it can need.
            allowance = budget - (estimate_text_tokens(_withheld_notice(len(owed))) + 1)
            kept = run_back(allowance)
        shown = lines[len(lines) - kept :]
        if shown and estimate_text_tokens(shown[-1]) + 1 > allowance:
            # Only the latest line can be over the allowance: `run_back` stops before any
            # other that would be.
            shown[-1] = _cut_to_fit(*said[-1], allowance)
        # Counted against every owed utterance, not against what happened to render: a
        # failed turn carries no content, is filtered out of `lines`, and is still
        # something this speaker never saw. Counting only rendered lines understated the
        # gap, in a notice whose whole purpose is to state its size.
        withheld = len(owed) - len(shown)
        if withheld > 0:
            shown = [_withheld_notice(withheld), *shown]
        return "\n".join(shown)


def _person_names(state: RoomState) -> tuple[str, ...]:
    """The names that mean one of the room's people: each one's id, display name and aliases.

    A name is left out when it could mean someone else in the room (#1868): another
    participant goes by the same name, or it is one word and another participant's name
    holds that word, as "Kim" does in "Kim Jiho". The person's id is held to the same rule
    (#1893): an id `kim` beside a clone shown as "Kim" is left out, so a fact about that
    clone keeps its subject. An id no one else in the room goes by always counts. Names are
    compared folded (`fold_name`, NFC included), and each is given once: a seat whose
    display name is its id, as a head's `user` is, gives `("user",)` (#1885).
    """
    names: list[str] = []
    given: set[str] = set()
    for person in state.participants:
        if person.kind is not ParticipantKind.HUMAN:
            continue
        _add_unshared_names(state, person, names, given)
    return tuple(names)


def _clone_names(state: RoomState, clone: Participant) -> tuple[str, ...]:
    """The names that mean `clone` itself in this room: its id, display name and aliases.

    Held to the rule `_person_names` holds the person's to, so a name another participant
    goes by, whole or as one word of theirs, is left out: a fact about that participant
    keeps its subject instead of becoming one about this clone (#2016).
    """
    names: list[str] = []
    _add_unshared_names(state, clone, names, set())
    return tuple(names)


def _add_unshared_names(
    state: RoomState, participant: Participant, names: list[str], given: set[str]
) -> None:
    """Append to `names` each of `participant`'s names no other participant could mean."""
    others = [p for p in state.participants if p.id != participant.id]
    other_names = {fold_name(n) for p in others for n in (p.id, p.display_name, *p.aliases)}
    other_words = {w for n in other_names for w in re.findall(r"\w+", n)}
    for name in (participant.id, participant.display_name, *participant.aliases):
        folded = fold_name(name)
        if folded in given:
            continue
        # A name of two words or more is never one of `other_words`.
        if not (folded in other_names or folded in other_words):
            given.add(folded)
            names.append(name)


def _lesson_lines(
    state: RoomState,
    speaker: Participant,
    since: int,
    rendered_through: int,
    content: str,
    executions: Sequence[ToolExecutionRecord],
) -> tuple[SpanLine, ...]:
    """What `speaker` was shown and did in one turn, each line labelled by its source.

    The span is the rows the turn rendered that the speaker had not seen (`since` to
    `rendered_through`): a person's are `person`, another clone's are `other`. Then the
    turn's successful tool results, except a story's contents and the memory tools' own
    echoes, and last what the speaker said. Only `person` and `tool` lines can ground a
    fact; the speaker's own words and another clone's are context for reading them.
    """
    kinds = {p.id: p.kind for p in state.participants}
    names = {p.id: p.display_name for p in state.participants}
    lines: list[SpanLine] = []
    for row in state.transcript:
        if not (since < row.seq <= rendered_through) or not row.is_utterance:
            continue
        if row.sender_id == speaker.id or not row.content.strip():
            continue
        kind = "person" if kinds.get(row.sender_id) is ParticipantKind.HUMAN else "other"
        lines.append(SpanLine(kind, names.get(row.sender_id, row.sender_id), row.content))
    for record in executions:
        if record.status is not ToolResultStatus.SUCCESS:
            continue
        if record.tool_name.startswith(EXCLUDED_TOOL_PREFIXES):
            continue
        output = record.output
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        if text.strip() and output is not None:
            lines.append(SpanLine("tool", record.tool_name, text))
    if content.strip():
        lines.append(SpanLine("self", speaker.display_name, content))
    return tuple(lines)


def _withheld_notice(withheld: int) -> str:
    """The line that opens a span when `withheld` earlier messages are not in it."""
    plural = "s" if withheld != 1 else ""
    return f"[...{withheld} earlier message{plural} in this room are not shown]"


#: Ends a line `_cut_to_fit` shortened, so the seat knows the message goes on.
_CUT_MARKER = "[...the rest of this message is not shown]"


def _cut_to_fit(prefix: str, content: str, allowance: int) -> str:
    """`prefix + content`, with `content` shortened so the line costs at most `allowance`.

    The sender prefix and the marker are kept whatever the allowance, so the line still
    says who wrote it and that it was cut. Cut on the UTF-8 bytes `estimate_text_tokens`
    counts, and never inside a character.
    """
    frame = estimate_text_tokens(f"{prefix} {_CUT_MARKER}")
    # One token for the newline, as `run_back` costs it; four bytes to a token.
    room = max(0, (allowance - 1 - frame) * 4)
    head = content.encode("utf-8")[:room].decode("utf-8", errors="ignore").rstrip()
    return f"{prefix}{head} {_CUT_MARKER}" if head else f"{prefix}{_CUT_MARKER}"


def span_token_budget(state: RoomState, agent: object) -> int:
    """The estimated tokens a turn's span may take (#1641).

    The smaller of the room's `RoomPolicy.max_span_tokens` and one `SPAN_WINDOW_SHARE`-th
    of the seat's context window. The window is the one the seat's own compaction counts
    against (`BaseAgent.context_window`), so a seat on a small local window is handed a
    span it can hold beside its prefix and history. A seat that reports no window -- one
    whose provider does not say, or an agent that is not a `BaseAgent` -- gets the
    policy's ceiling alone.
    """
    ceiling = state.policy.max_span_tokens
    read_window = getattr(agent, "context_window", None)
    window = read_window() if callable(read_window) else None
    if isinstance(window, int) and not isinstance(window, bool) and window > 0:
        return max(1, min(ceiling, window // SPAN_WINDOW_SHARE))
    return ceiling
