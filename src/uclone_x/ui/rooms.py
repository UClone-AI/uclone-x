"""The head's room surface: `/api/rooms`.

Kept out of `app.py` because that module is already the largest in the package and this
is a self-contained slice -- one stack of Core objects and the routes over them.

**What this layer is allowed to decide.** Nothing about conversations. The Core owns the
transcript, the roster, the ceiling and the selector chain; this module translates HTTP
into those calls and Core refusals back into status codes that keep their reason. The one
judgement it does make is *when to answer*: sending a message answers 202 and drives the
cascade behind the response, because `RoomOrchestrator.post` returns only once every turn
has landed and a route that awaited it would hold one request across several model calls.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from uclone_x.agent.composition import MissingCapabilityError
from uclone_x.core.session_diagnostics import DEFAULT_MAX_CONVERSATION_TURNS
from uclone_x.errors import (
    BudgetExceededError,
    NothingToRetryError,
    ParticipantNotResolvableError,
    RoomAlreadyExistsError,
    RoomError,
    RoomNotFoundError,
    SecondHumanInRoomError,
    SessionIdCollisionError,
    SessionMutationDuringTurnError,
    StaleRoomWriteError,
    TokenBudgetExhaustedError,
    TurnNotLandedError,
    TurnNotStartedError,
    UnknownRoomParticipantError,
    UnreadableRoomRecordError,
)
from uclone_x.llm.context_window import OLLAMA_CONTEXT_WINDOWS, published_context_window
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.room.knowledge import SEAT_KNOWLEDGE_SUBDIR
from uclone_x.room.knowledge_store import SeatKnowledgeStore
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomMessageKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
)
from uclone_x.room.orchestrator import (
    AUTONOMOUS_CIRCUIT_BREAKER_TURNS,
    RoomOrchestrator,
)
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import build_selector_chain
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore

if TYPE_CHECKING:  # pragma: no cover - import cycle; the app imports this module
    from uclone_x.agent.base import BaseAgent
    from uclone_x.llm.protocols import LLMProviderProtocol
    from uclone_x.ui.app import AgentSessionManager

__all__ = ["RoomStack", "register_room_routes", "seated_agents"]

logger = logging.getLogger(__name__)

#: The human a head seats when the caller names nobody. A room seats exactly one human
#: (#763), so this is a name for *the* operator rather than a default among several.
#: Used only at creation; it is never substituted for a *missing* participant, which is
#: how a room with nobody seated used to answer typing and sending in a stranger's name.
DEFAULT_HUMAN_ID = "user"

#: Active turns at which a seat is reported as saturated. The same figure `/api/chat`'s
#: readout used, taken from the Core's own default rather than restated here: two surfaces
#: over one runtime disagreeing about when a conversation is full is a worse defect than
#: either threshold being wrong.
SATURATION_TURNS_THRESHOLD = DEFAULT_MAX_CONVERSATION_TURNS

_KIND_REFUSAL = (
    "A participant is either an 'agent' or a 'human'; any other kind is refused rather "
    "than defaulted. Defaulting seated a person as an agent, which derives them a "
    "session, leaves the room with no human, and makes every later message unanswerable."
)
_NO_HUMAN_REFUSAL = (
    "Nobody is in this conversation. A room seats one human and this one seats none, so "
    "there is no sender to record. Add yourself to the conversation first."
)


def _required_str(req: Mapping[str, Any], field: str, *, allow_blank: bool = False) -> str:
    """The field as a string, refusing anything that is not one.

    `str(req.get(field, ""))` looked equivalent and was not: it turns `None` into the
    string `"None"` and a mapping into its Python repr, so the Core's own refusals --
    a blank title above all -- never fired and the head supplied a value nobody typed.

    `allow_blank` hands a whitespace-only value through to the Core, whose own refusal is
    the better one to show: it says what a title is *for*. The head still refuses a blank
    where no Core refusal exists -- an empty message would otherwise be recorded and
    answered, spending a turn on nothing.

    Raises:
        HTTPException: 400, naming the field and what was wrong with it.
    """
    value = req.get(field)
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field!r} must be text; got {type(value).__name__}. A value coerced to "
                f"text here would be stored and shown as whatever it stringified to."
            ),
        )
    if not allow_blank and not value.strip():
        raise HTTPException(status_code=400, detail=f"{field!r} must not be empty.")
    return value


def _refuse_an_unaddressable_id(participant_id: str) -> None:
    """Refuse an id that no route could name again.

    Participants are addressed as one path segment, so an id carrying a separator or an
    escape cannot be removed once seated -- and since a room seats exactly one human and
    removal is the only way to free the seat, such an id makes the room permanently
    unusable. The Core validates agent ids on the derivation it needs them for; this is
    the head's own obligation, because URL addressability is the head's problem.

    Raises:
        HTTPException: 400, naming the character.
    """
    for bad in ("/", "\\", "%", "?", "#"):
        if bad in participant_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A participant id must not contain {bad!r}: participants are "
                    f"addressed as one path segment, and an id holding it could be "
                    f"seated and never removed."
                ),
            )


@dataclass(frozen=True, slots=True)
class _RoomRuntime:
    """One room's live machinery: the orchestrator driving it and the resolver it drives.

    Held together in one entry rather than in two maps keyed the same way, because the
    two are created together and must die together. A `forget` that dropped only the
    orchestrator would leave the resolver's cached agents behind, and this class is read
    for *which clones are running* -- so the surviving half would keep reporting a
    deleted room's agents as live, with nothing left that could correct it.
    """

    orchestrator: RoomOrchestrator
    resolver: RoomAgentResolver


class RoomStack:
    """The Core objects a room route needs, assembled once and kept.

    One orchestrator **per room**, cached. Not one per request: `interrupt` cancels the
    turn task the orchestrator is holding, so a Stop served by a freshly built instance
    would find no turn and silently do nothing. Not one for all rooms either: the selector
    chain is built from the room's own policy, and a single chain would route every room
    by whichever policy happened to be read first.

    The resolver is shared and long-lived on purpose -- it caches one agent per
    `(room, participant)` session, and rebuilding it per request would reload every
    agent's session from disk on every turn.
    """

    def __init__(self, session_mgr: AgentSessionManager) -> None:
        self._session_mgr = session_mgr
        self.store = RoomStore(session_mgr.storage_dir / "rooms")
        self.service = RoomService(self.store)
        #: Every seat's knowledge, written by each room's orchestrator after a turn and
        #: loaded by its resolver before the seat's first one (#1367). One store for all
        #: rooms: a file is named by the seat's session id, which is already unique per
        #: room and seat. Public because the knowledge read answers from it when the seat
        #: is not running, without building an agent.
        self.knowledge = SeatKnowledgeStore(
            session_mgr.storage_dir / SEAT_KNOWLEDGE_SUBDIR,
            engine_factory=lambda namespace: OntologyEngine(namespace_iri=namespace),
        )
        self._rooms: dict[str, _RoomRuntime] = {}
        #: Live cascades. A *set* per room, not one task: a second send used to overwrite
        #: the entry and leave the first cascade running untracked, so neither a delete
        #: nor a shutdown could reach it. Cancelled by `forget` and by `close`.
        self._running: dict[str, set[asyncio.Task[Any]]] = {}
        #: Live recurring room loops: room_id -> (job_id, task, interval_seconds, prompt)
        self._room_loops: dict[str, tuple[str, asyncio.Task[None], float, str]] = {}
        # A conversation's seats and its routing model are built once per room, from the
        # connector of the moment. Without this a model chosen in Settings reached new
        # conversations only, while Settings said it had been applied (#1446).
        session_mgr.on_llm_replaced(self._llm_replaced)

    def _llm_replaced(self, llm: LLMProviderProtocol | None) -> None:
        """Hand every open conversation the connector Settings just installed."""
        for room_id, runtime in self._rooms.items():
            runtime.resolver.replace_llm(llm)
            try:
                state = self.store.load(room_id)
            except UnreadableRoomRecordError:
                # Its turns refuse on the same read, so its routing has nothing to answer;
                # failing here would fail the Settings save for every other conversation.
                logger.warning("Room %s kept its routing model: its record will not load", room_id)
                continue
            if state is not None:
                runtime.orchestrator.replace_selectors(
                    build_selector_chain(state.policy, provider=llm)
                )

    def _host(self) -> Any:
        from uclone_x.agent.composition import HostDependencies

        mgr = self._session_mgr
        tool_scoper = None
        settings = mgr.get_settings()
        provider = str(settings.get("llm_provider") or "").strip().lower()
        llm_base_url = str(settings.get("llm_base_url") or "").strip()
        if provider == "ollama" and llm_base_url:
            try:
                from uclone_x.llm.connectors.ollama_embedder import OllamaEmbedder
                from uclone_x.tools.tool_scoper import SemanticToolScoper

                embedder = OllamaEmbedder(base_url=llm_base_url)
                reg = mgr.skill_registry

                def skills_prov() -> list[tuple[str, str]]:
                    return [(s.manifest.name, s.manifest.description) for s in reg.list_skills()]

                tool_scoper = SemanticToolScoper(
                    embedder,
                    top_k=5,
                    threshold=0.30,
                    always_include=("record_memory_fact",),
                    skills_provider=skills_prov,
                )
            except Exception:
                from uclone_x.tools.tool_scoper import LexicalToolScoper

                tool_scoper = LexicalToolScoper(top_k=5)
        else:
            from uclone_x.tools.tool_scoper import LexicalToolScoper

            tool_scoper = LexicalToolScoper(top_k=5)

        return HostDependencies(
            bus=mgr.bus,
            llm=mgr.llm,
            tools=mgr.tools,
            tracer=mgr.tracer,
            store=mgr.core_store,
            budget=mgr.budget_tracker,
            skills=mgr.skill_registry,
            ontology=mgr.ontology_engine,
            tool_scoper=tool_scoper,
        )

    def orchestrator(self, state: RoomState) -> RoomOrchestrator:
        """The orchestrator driving this room, built on first use from its policy."""
        existing = self._rooms.get(state.room_id)
        if existing is not None:
            return existing.orchestrator
        resolver = RoomAgentResolver(
            self._host(),
            # Each participant induces into its own graph (P7, G4). Without this the
            # resolver refuses every seated agent, because the host carries one shared
            # engine and handing it to all of them merges what each learned separately.
            ontology_factory=lambda namespace: OntologyEngine(namespace_iri=namespace),
            # ...and each keeps its own memory. Without this the room's agents are
            # composed with no store, so `record_memory_fact` is neither advertised to
            # them nor resolvable: a room agent could not remember anything at all.
            #
            # The session manager's map, not one of this class's own. A store is per agent
            # id, and an id names one file: a second map here would hand `champion` in a
            # room a different object than `champion` in chat, and since `save()` rewrites
            # the whole document each would drop what the other recorded, with nothing
            # reporting it. One resolver per room also rules out holding it in the
            # resolver, which is what makes the manager's the only correct home.
            memory_factory=self._session_mgr.memory_for,
            workspace_root=self._session_mgr.workspace_dir,
            read_roots=lambda: self._session_mgr.read_roots,
            knowledge=self.knowledge,  # read before a seat's first turn (#1367)
        )
        built = RoomOrchestrator(
            store=self.store,
            selectors=build_selector_chain(state.policy, provider=self._session_mgr.llm),
            resolver=resolver,
            bus=self._session_mgr.bus,
            knowledge=self.knowledge,  # written after each turn (#1367)
        )
        self._rooms[state.room_id] = _RoomRuntime(orchestrator=built, resolver=resolver)
        return built

    def turn_in_flight(self, room_id: str) -> bool:
        """Whether a cascade this stack started is still running in this room.

        Public because it is a *refusal's* input, not an internal detail: the history
        controls decline while a turn is running, and a refusal reachable only through
        private state is one no test can state. Read from `_running` rather than from the
        orchestrator's own floor, because this is the question the route is asking — a
        cascade between turns holds no floor and is still about to write.
        """
        return any(not task.done() for task in self._running.get(room_id, set()))

    def turn_unlanded(self, room_id: str) -> bool:
        """Whether this room's orchestrator has a turn counted as started and not yet saved.

        Separate from `turn_in_flight`, which asks about a cascade: a retry runs its turn
        inside the request rather than as a cascade, and still has a turn in progress. The
        dock subtracts this one turn from the started-minus-landed difference; whatever is
        left is a turn whose result was lost (#1366). Never builds an orchestrator.
        """
        runtime = self._rooms.get(room_id)
        return runtime is not None and runtime.orchestrator.turn_unlanded(room_id)

    def note_presence(self, state: RoomState, active: bool = True) -> None:
        """Report presence for a room, ensuring its orchestrator is built."""
        orch = self.orchestrator(state)
        orch.note_presence(state.room_id, active)

    def is_presence_active(self, room_id: str) -> bool:
        """Whether the user is currently actively viewing the room."""
        runtime = self._rooms.get(room_id)
        return runtime is not None and runtime.orchestrator.is_presence_active(room_id)

    async def set_autonomous(self, state: RoomState, enabled: bool) -> RoomState:
        """Toggle autonomous discussion on room policy."""
        updated_policy = state.policy.model_copy(update={"autonomous": enabled})
        turn_state = state.turn_state
        if enabled and turn_state.agent_turns_since_human >= AUTONOMOUS_CIRCUIT_BREAKER_TURNS:
            turn_state = turn_state.model_copy(update={"agent_turns_since_human": 0})
        saved = self.store.save(
            state.model_copy(update={"policy": updated_policy, "turn_state": turn_state})
        )
        orch = self.orchestrator(saved)
        orch.note_presence(state.room_id, active=True)
        if enabled and not self.turn_in_flight(state.room_id):
            last_seq = saved.transcript[-1].seq if saved.transcript else 0
            self.drive(state.room_id, lambda: orch.resume(state.room_id, last_seq))
        return saved

    def live_agent(self, room_id: str, session_id: str) -> BaseAgent | None:
        """The agent this room's resolver has already built for a seat, or `None`.

        `None` means nobody has spoken in that seat since the process started, not that the
        seat is empty. Deliberately does not construct one: see `RoomAgentResolver.live_agent`.

        Cast, because the resolver holds its agents as `BaseAgentProtocol` while every one it
        builds comes from `compose_agent`, whose return type is `BaseAgent`. The two history
        controls need `get_session`, which the protocol does not declare — widening the
        protocol for two callers would oblige every fake in the tree to grow a method neither
        the orchestrator nor the selectors ever call.
        """
        runtime = self._rooms.get(room_id)
        if runtime is None:
            return None
        return cast("BaseAgent | None", runtime.resolver.live_agent(session_id))

    async def resolve_agent(self, state: RoomState, participant: Participant) -> BaseAgent:
        """The agent for a seat, built if this is the first time it is needed.

        Used by compaction, which cannot be done to a seat that does not exist yet, and not
        by the reads — asking how full a context is must not be the act that fills it.
        """
        self.orchestrator(state)
        runtime = self._rooms[state.room_id]
        return cast("BaseAgent", await runtime.resolver.resolve(participant))

    def session_manager(self) -> AgentSessionManager:
        """The manager that owns session records, for the routes that cut them back."""
        return self._session_mgr

    def seated_agent_ids(self) -> frozenset[str]:
        """Every agent id with a live instance seated in some room right now.

        The half of "which clones are running" that `AgentSessionManager.list_agents()`
        cannot see: a clone taking part in a conversation is built and cached by that
        room's resolver, and never written back into the manager's chat map. Under D1
        (design §1.3) every new conversation is a room, so this is the ordinary case and
        not an edge one.

        Union, not a per-room answer, because a clone is running if it is running
        anywhere; and unioned over the live rooms only, so a deleted room's agents stop
        counting the moment `forget` drops its runtime.
        """
        seated: set[str] = set()
        for runtime in self._rooms.values():
            seated |= runtime.resolver.seated_agent_ids()
        return frozenset(seated)

    def schedule_room_loop(
        self,
        room_id: str,
        sender_id: str,
        interval_seconds: float,
        prompt: str,
    ) -> str:
        """Schedule a recurring prompt execution in the room."""
        self.cancel_room_loop(room_id)
        job_id = f"loop-{uuid.uuid4().hex[:6]}"

        async def _loop_worker() -> None:
            try:
                first = True
                while True:
                    if not first:
                        await asyncio.sleep(interval_seconds)
                    first = False

                    state = self.store.load(room_id)
                    if state is None:
                        break
                    orch = self.orchestrator(state)
                    try:
                        state = await orch.accept(room_id, sender_id, prompt)
                        seq = state.transcript[-1].seq
                        await orch.resume(room_id, seq)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("Room loop tick encountered error for %s: %s", room_id, exc)
            except asyncio.CancelledError:
                pass
            finally:
                current = self._room_loops.get(room_id)
                if current is not None and current[0] == job_id:
                    self._room_loops.pop(room_id, None)

        task = asyncio.create_task(_loop_worker(), name=f"room-loop-{room_id}-{job_id}")
        self._room_loops[room_id] = (job_id, task, interval_seconds, prompt)
        return job_id

    def cancel_room_loop(self, room_id: str) -> bool:
        """Cancel any active recurring loop for this room."""
        item = self._room_loops.pop(room_id, None)
        if item is not None:
            _, task, _, _ = item
            if not task.done():
                task.cancel()
            return True
        return False

    def get_room_loop_info(self, room_id: str) -> tuple[str, float, str] | None:
        """Return (job_id, interval_seconds, prompt) if an active loop is running."""
        item = self._room_loops.get(room_id)
        if item is not None and not item[1].done():
            return (item[0], item[2], item[3])
        return None

    def forget(self, room_id: str) -> None:
        """Drop a deleted room's orchestrator, and every turn it may still be running."""
        self.cancel_room_loop(room_id)
        self._rooms.pop(room_id, None)
        for task in self._running.pop(room_id, set()):
            if not task.done():
                task.cancel()

    async def close(self) -> None:
        """Cancel every live cascade and wait for it, on the way down.

        The comment on `_running` used to claim a shutdown did this and nothing called
        it. A turn killed at interpreter teardown between building its `RoomMessage` and
        saving it spends the turn, pays for the tokens, and records nothing -- leaving a
        room whose `last_decision` names a speaker with no utterance.
        """
        for room_id in list(self._room_loops.keys()):
            self.cancel_room_loop(room_id)
        tasks = [task for group in self._running.values() for task in group]
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown path
                pass
        self._running.clear()

    def drive(self, room_id: str, start: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Run a cascade behind an answered request, and never lose its failure.

        A background task whose exception nobody retrieves is a room that stopped for a
        reason printed only at interpreter exit. The failure is logged here and announced
        on the room's own topic, so the surface can say the conversation stopped and why
        rather than showing a speaker that writes forever.
        """

        async def run() -> None:
            try:
                await start()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Room %s stopped mid-cascade: %s", room_id, exc, exc_info=True)
                reason = reader_facing_reason(exc)
                await self._announce_failure(room_id, reason)
                try:
                    failed_state = self.store.load(room_id)
                    if failed_state is not None:
                        err_decision = SpeakerDecision(
                            verdict=SelectionVerdict.SILENCE,
                            selector="orchestrator",
                            reasoning=f"cascade stopped: {reason}",
                        )
                        self.store.save(
                            failed_state.model_copy(update={"last_decision": err_decision})
                        )
                except Exception:
                    logger.debug(
                        "Could not record cascade failure decision for room %s",
                        room_id,
                        exc_info=True,
                    )
            finally:
                group = self._running.get(room_id)
                if group is not None:
                    group.discard(task)
                    if not group:
                        self._running.pop(room_id, None)

        # A coroutine built at the call site and never awaited -- which a cancel before
        # the first step produced -- left the human's message recorded and no turn ever
        # run: a room that says "thinking" forever. The factory is called inside the task.
        task = asyncio.create_task(run())
        self._running.setdefault(room_id, set()).add(task)

    async def _announce_failure(self, room_id: str, error: str) -> None:
        """Put a cascade-level failure on `room.{room_id}` as a fourth reply status.

        The topic already carries `generating`, `streaming` and `final`; this is the case
        where no agent ever reached any of them -- an unresolvable address, a resolver
        refusal, a store conflict. It carries no `agent_id` because nobody spoke.

        `error` is the reader-facing reason from `reader_facing_reason`, not the raw
        exception text. This channel is forwarded by `/api/stream` to every subscriber and
        rendered in the conversation, so a `RoomError`'s own sentence belongs here and an
        unexpected fault's `repr` -- which can carry a store path -- does not.
        """
        from uclone_x.engine.event_bus import AgentEvent, EventSource, EventType

        bus = self._session_mgr.bus
        try:
            publisher = bus.register_publisher(sender_id="room", source=EventSource.SYSTEM)
            await publisher.publish(
                AgentEvent(
                    type=EventType.AGENT_REPLY,
                    topic=f"room.{room_id}",
                    payload={"room_id": room_id, "status": "error", "error": error},
                )
            )
        except Exception:
            # `warning`, not `debug`: this notice is how the surface learns the conversation
            # stopped, so losing it leaves a speaker that writes forever -- a failure in its
            # own right, and one that must be findable without a debug log level (#929).
            logger.warning("Room %s could not announce its failure", room_id, exc_info=True)


def reader_facing_reason(exc: Exception) -> str:
    """What a person may be shown about a failure, as against what the log records.

    Public, and named without the underscore for that reason: which half of a failure a
    reader may see is a rule worth a test of its own, and a rule reachable only through
    private state is not an interface.

    A Core refusal is written for the person who caused it -- `RoomError`'s messages say
    what is wrong and what would be allowed -- so it is passed through. Anything else is a
    fault, and its text is ours: an `f"{type(exc).__name__}: {exc}"` was reaching the
    conversation as copy, which put a Python class name and whatever the exception happened
    to interpolate (a store path, for instance) in front of a non-expert reader.
    """
    if isinstance(exc, RoomError):
        return str(exc)
    if isinstance(exc, MissingCapabilityError) and "llm" in exc.missing:
        # The one missing capability that is the reader's to supply. Reported as "a
        # problem in the agent runtime" before #1446, which named neither the cause nor
        # where to fix it.
        return NO_MODEL_REASON
    return "This conversation stopped because of a problem in the agent runtime."


#: What a conversation says when a seat cannot be built because no model is selected.
NO_MODEL_REASON = (
    "No model is selected, so nobody in this conversation can answer. "
    "Choose a model in Settings, then send your message again."
)


#: What `/api/rooms/{id}` answers when the room's own record will not load (#1411). Plain,
#: because the head shows a `detail` as the Core's words: what happened, and where the
#: reason went. The parser's field dump is for the log.
UNREADABLE_ROOM_DETAIL = (
    "This conversation's saved copy could not be read, so it cannot be opened or changed. "
    "The reason is in the server log."
)


def _http_error(exc: Exception) -> HTTPException:
    """Translate a Core refusal into a status code that keeps the refusal's own words.

    The message is the product of the refusal, not decoration on it: `RoomPolicy` names
    the two knobs that disagree, the roster says why a second human is refused, and a
    generic 422 with a validation dump throws all of that away -- leaving a surface that
    can only say "invalid".
    """
    if isinstance(exc, HTTPException):
        # Already an answer. The call sites wrap broad `except Exception`, so a refusal
        # raised by a helper inside one of them arrived here and was re-dressed as a 500.
        return exc
    if isinstance(exc, RoomNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TurnNotStartedError):
        # The room could not record the turn's start, so it refused to run it: a store
        # fault, not the caller's, and asking again may succeed.
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, TurnNotLandedError):
        # The same store fault at the other end of the turn: the reply was not kept, the
        # seat was put back, and asking again may succeed (#1495).
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (RoomAlreadyExistsError, SecondHumanInRoomError, StaleRoomWriteError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, NothingToRetryError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (UnknownRoomParticipantError, ParticipantNotResolvableError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RoomError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (BudgetExceededError, TokenBudgetExhaustedError)):
        # A refusal, not a fault: the room asked for a turn the budget will not pay for.
        return HTTPException(status_code=402, detail=str(exc))
    if isinstance(exc, SessionIdCollisionError):
        # A room route reaches the session store through the seats -- `/context` reads
        # every seat's record -- so it can meet a refusal the Core raises about a
        # *session* rather than about the room. The status is not decided here: #256
        # settled it at 409 in `ui/app._translate_session_error`, and a second surface
        # answering 400 or 500 for the same refusal is the divergence P6 forbids. Without
        # this branch the fall-through below answers 500 with a message of our own, which
        # throws away the one sentence that says which two ids fold together (#1212).
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SessionMutationDuringTurnError):
        # The refusal that actually covers `/compact` (#1213), and it was arriving as a
        # bare 500. `_refuse_during_turn` cannot be compaction's guard -- see its
        # docstring -- so the Core's own per-seat check is, and a guard whose answer is
        # "the conversation service failed, see the server log" is one no caller can act
        # on: the Core's sentence names the seat, the session and the remedy ("await the
        # turn first"), and none of it reached the browser. 409, because that is what
        # `ui/app._translate_session_error` already answers for this exact class; a
        # second surface answering 500 for the same refusal is the divergence the
        # collision branch above names (#1212, #256).
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, MissingCapabilityError) and "llm" in exc.missing:
        # A retry builds its seat inside the request, so it meets the same missing model a
        # cascade reports on the room's topic, and says it the same way (#1446).
        return HTTPException(status_code=409, detail=NO_MODEL_REASON)
    if isinstance(exc, UnreadableRoomRecordError):
        # A record the store wrote, and not the caller's request: a fault, answered in our
        # words, with the parser's cause in the log. Before #1411 the store raised the bare
        # `ValidationError`, which the branch below answered as a 400 whose reason was the
        # field dump (`transcript.3.kind: Input should be ...`) -- shown, through the
        # head's "the Core's own words" rule, to someone who does not read either.
        logger.error("Room %r could not be loaded from its record", exc.room_id, exc_info=exc)
        return HTTPException(status_code=500, detail=UNREADABLE_ROOM_DETAIL)
    if isinstance(exc, ValidationError):
        # One line per offending field, in the model's own words. For a *request* the
        # caller sent; a stored record that will not load is the branch above.
        return HTTPException(
            status_code=400,
            detail="; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            ),
        )
    # Anything left is a fault rather than a refusal, and its message is ours, not the
    # caller's: an `AttributeError`'s text was being returned to the browser verbatim.
    logger.error("Unhandled error on a room route: %s", exc, exc_info=True)
    return HTTPException(
        status_code=500,
        detail="The conversation service failed. The reason is in the server log.",
    )


def _summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "room_id": summary.room_id,
        "title": summary.title,
        "agent_ids": list(summary.agent_ids),
        "human_ids": list(summary.human_ids),
        "message_count": summary.message_count,
        "updated_at": summary.updated_at,
    }


def register_room_routes(app: FastAPI, stack: RoomStack) -> None:
    """Mount `/api/rooms` on `app`.

    There is deliberately **no** `/api/rooms/{id}/events`. `/api/stream` already
    subscribes to every topic and forwards each event's own `topic`, so a room's events
    reach the browser over the connection that is already open. A second SSE generator
    would have to repeat its shutdown and drain handling and would cost one EventSource
    per open room.
    """
    service = stack.service

    def _room(room_id: str) -> RoomState:
        try:
            return service.get(room_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/rooms")
    async def list_rooms() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every stored conversation, by title rather than by id.

        `unreadable` names the records that are there and will not load (#1440), so the
        head can still show them and offer to delete them.
        """
        listing = service.survey_rooms()
        return {
            "rooms": [_summary_payload(s) for s in listing.rooms],
            "unreadable": list(listing.unreadable),
        }

    @app.post("/api/rooms", status_code=201)
    async def create_room(req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Create a conversation and seat its roster.

        Two writes: `RoomService.create` takes a title and nothing else, and participants
        arrive through `add_participant`. If the second fails the first is undone -- a
        titled, empty room left in the list is indistinguishable from one somebody meant
        to make, and the user is given no way to tell.
        """
        try:
            policy = RoomPolicy.model_validate(req.get("policy") or {})
        except ValidationError as exc:
            raise _http_error(exc) from exc
        # Blank is the Core's refusal to make, and its wording is the one to show.
        title = _required_str(req, "title", allow_blank=True)
        human_id = req.get("human_id") or DEFAULT_HUMAN_ID
        if not isinstance(human_id, str):
            raise HTTPException(status_code=400, detail="'human_id' must be text.")
        # Checked before anything is written, so a bad id is a refusal rather than a
        # rollback.
        _refuse_an_unaddressable_id(human_id)
        try:
            state = service.create(title, policy=policy)
        except Exception as exc:
            raise _http_error(exc) from exc
        raw_agents: object = req.get("agent_ids") or []
        agent_ids: list[str] = (
            [str(item) for item in cast(list[object], raw_agents)]
            if isinstance(raw_agents, list)
            else []
        )
        try:
            state = service.add_participant(state.room_id, human_id, kind=ParticipantKind.HUMAN)
            for agent_id in agent_ids:
                state = service.add_participant(state.room_id, agent_id)
        except Exception as exc:
            _roll_back(service, state.room_id, exc)
            raise _http_error(exc) from exc
        return state.model_dump(mode="json")

    @app.get("/api/rooms/{room_id}")
    async def get_room(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """One conversation, roster and transcript included."""
        return _room(room_id).model_dump(mode="json")

    @app.patch("/api/rooms/{room_id}")
    async def rename_room(room_id: str, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Give a conversation a new title, carrying creation's refusals."""
        try:
            return service.rename(
                room_id, _required_str(req, "title", allow_blank=True)
            ).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/rooms/{room_id}", status_code=204)
    async def delete_room(room_id: str) -> None:  # pyright: ignore[reportUnusedFunction]
        """Remove a conversation, and stop anything still running in it.

        A record that is there and will not load is still removed (#1440). Deleting is the
        one thing a reader can do with it, and it needs nothing from inside the record.
        """
        try:
            service.get(room_id)
        except UnreadableRoomRecordError:  # deleted anyway (#1440)
            logger.warning("Deleting room %r, whose record will not load", room_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        stack.forget(room_id)
        try:
            service.delete(room_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/rooms/{room_id}/participants")
    async def add_participant(room_id: str, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Seat somebody in a conversation that is already running."""
        _room(room_id)
        raw_id = req.get("agent_id") or req.get("participant_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise HTTPException(status_code=400, detail="'agent_id' must be text.")
        participant_id = raw_id
        _refuse_an_unaddressable_id(participant_id)
        raw_kind = req.get("kind", "agent")
        if raw_kind not in ("agent", "human"):
            raise HTTPException(status_code=400, detail=_KIND_REFUSAL)
        kind = ParticipantKind.HUMAN if raw_kind == "human" else ParticipantKind.AGENT
        try:
            state = service.add_participant(room_id, participant_id, kind=kind)
        except Exception as exc:
            raise _http_error(exc) from exc
        # The roster decides the chain, so the cached one is out of date.
        return state.model_dump(mode="json")

    @app.delete("/api/rooms/{room_id}/participants/{participant_id}")
    async def remove_participant(room_id: str, participant_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Remove somebody from a conversation."""
        _room(room_id)
        try:
            state = service.remove_participant(room_id, participant_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return state.model_dump(mode="json")

    @app.post("/api/rooms/{room_id}/messages", status_code=202)
    async def send_message(room_id: str, req: dict[str, Any]) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Record a message and answer; the turns it causes run behind the response.

        202 and not the final room: `post()` spans every turn of the cascade, so awaiting
        it holds one HTTP request across several model calls -- a proxy timeout on a long
        exchange, and unabortable while it runs. Every refusal is still synchronous: the
        message is recorded by `accept` before this answers, so an unknown sender is an
        answer to the request and not an event on a topic nobody may be reading.
        """
        state = _room(room_id)
        orch = stack.orchestrator(state)
        content = _required_str(req, "content")
        raw_sender = req.get("sender_id")
        sender_id = raw_sender if isinstance(raw_sender, str) and raw_sender else _sole_human(state)
        clean_content = content.strip()
        if clean_content.startswith("/loop"):
            from uclone_x.agent.loop.parser import parse_loop_command_input

            try:
                state = await orch.accept(room_id, sender_id, content)
            except Exception as exc:
                raise _http_error(exc) from exc

            if clean_content in ("/loop", "/loop help"):
                help_msg = (
                    "ℹ️ **`/loop` 명령어 안내:**\n\n"
                    "• `/loop <간격> <프롬프트>`: 주기적으로 프롬프트 실행 (예: `/loop 1분마다 하나씩 만들어보자` 또는 `/loop 30s 상태 확인`)\n"
                    "• `/loop list`: 현재 대화방의 활성 반복 작업 확인\n"
                    "• `/loop stop`: 현재 대화방의 반복 작업 중지 (상단 중단 버튼으로도 중지 가능)"
                )
                note = RoomMessage(
                    seq=len(state.transcript) + 1,
                    sender_id="system",
                    content=help_msg,
                    kind=RoomMessageKind.UTTERANCE,
                )
                state = stack.store.save(
                    state.model_copy(update={"transcript": (*state.transcript, note)})
                )
                return JSONResponse(
                    status_code=202, content={"room_id": room_id, "seq": state.transcript[-1].seq}
                )

            if clean_content == "/loop list":
                loop_info = stack.get_room_loop_info(room_id)
                if loop_info is not None:
                    job_id, interval, prompt_text = loop_info
                    intvl_display = (
                        f"{int(interval)}초" if interval < 60 else f"{int(interval // 60)}분"
                    )
                    list_msg = (
                        f"🔄 **활성 반복 작업:** `{job_id}` ({intvl_display} 주기)\n"
                        f'프롬프트: "{prompt_text}"'
                    )
                else:
                    list_msg = "ℹ️ 현재 이 대화방에 실행 중인 반복 작업이 없습니다."
                note = RoomMessage(
                    seq=len(state.transcript) + 1,
                    sender_id="system",
                    content=list_msg,
                    kind=RoomMessageKind.UTTERANCE,
                )
                state = stack.store.save(
                    state.model_copy(update={"transcript": (*state.transcript, note)})
                )
                return JSONResponse(
                    status_code=202, content={"room_id": room_id, "seq": state.transcript[-1].seq}
                )

            if clean_content in ("/loop stop", "/loop stop all") or clean_content.startswith(
                "/loop stop "
            ):
                stopped = stack.cancel_room_loop(room_id)
                stop_msg = (
                    "🛑 반복 실행 작업이 중지되었습니다."
                    if stopped
                    else "ℹ️ 중지할 활성 반복 작업이 없습니다."
                )
                note = RoomMessage(
                    seq=len(state.transcript) + 1,
                    sender_id="system",
                    content=stop_msg,
                    kind=RoomMessageKind.UTTERANCE,
                )
                state = stack.store.save(
                    state.model_copy(update={"transcript": (*state.transcript, note)})
                )
                return JSONResponse(
                    status_code=202, content={"room_id": room_id, "seq": state.transcript[-1].seq}
                )

            if clean_content.startswith("/loop "):
                raw_arg = clean_content[len("/loop ") :].strip()
                try:
                    interval_seconds, clean_prompt = parse_loop_command_input(raw_arg)
                except ValueError as err:
                    err_msg = f"⚠️ `/loop` 형식 오류: {err}"
                    note = RoomMessage(
                        seq=len(state.transcript) + 1,
                        sender_id="system",
                        content=err_msg,
                        kind=RoomMessageKind.UTTERANCE,
                    )
                    state = stack.store.save(
                        state.model_copy(update={"transcript": (*state.transcript, note)})
                    )
                    return JSONResponse(
                        status_code=202,
                        content={"room_id": room_id, "seq": state.transcript[-1].seq},
                    )

                job_id = stack.schedule_room_loop(
                    room_id=room_id,
                    sender_id=sender_id,
                    interval_seconds=interval_seconds,
                    prompt=clean_prompt,
                )
                intvl_display = (
                    f"{int(interval_seconds)}초"
                    if interval_seconds < 60
                    else f"{int(interval_seconds // 60)}분"
                )
                ack_msg = (
                    f"🔄 **반복 작업 등록됨** (매 {intvl_display}마다 실행, ID: `{job_id}`):\n"
                    f'"{clean_prompt}"\n\n'
                    f"*중지하려면 `/loop stop`을 입력하거나 상단 대화 중단 버튼을 누르세요.*"
                )
                note = RoomMessage(
                    seq=len(state.transcript) + 1,
                    sender_id="system",
                    content=ack_msg,
                    kind=RoomMessageKind.UTTERANCE,
                )
                state = stack.store.save(
                    state.model_copy(update={"transcript": (*state.transcript, note)})
                )
                return JSONResponse(
                    status_code=202, content={"room_id": room_id, "seq": state.transcript[-1].seq}
                )

        try:
            state = await orch.accept(room_id, sender_id, content)
        except Exception as exc:
            raise _http_error(exc) from exc
        seq = state.transcript[-1].seq
        stack.drive(room_id, lambda: orch.resume(room_id, seq))
        return JSONResponse(status_code=202, content={"room_id": room_id, "seq": seq})

    @app.post("/api/rooms/{room_id}/stop")
    async def stop_room(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Take the floor back. The invariant without this control is a claim."""
        stack.cancel_room_loop(room_id)
        state = _room(room_id)
        try:
            await stack.orchestrator(state).interrupt(room_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return _room(room_id).model_dump(mode="json")

    @app.post("/api/rooms/{room_id}/typing", status_code=204)
    async def report_typing(room_id: str) -> None:  # pyright: ignore[reportUnusedFunction]
        """Report that the operator is composing -- a timestamp, never keystrokes."""
        state = _room(room_id)
        try:
            await stack.orchestrator(state).note_human_activity(room_id, _sole_human(state))
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/api/rooms/{room_id}/autonomous")
    async def toggle_autonomous(room_id: str, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Toggle autonomous discussion mode for this room."""
        state = _room(room_id)
        enabled = bool(req.get("enabled", True))
        try:
            updated = await stack.set_autonomous(state, enabled)
        except Exception as exc:
            raise _http_error(exc) from exc
        return updated.model_dump(mode="json")

    @app.post("/api/rooms/{room_id}/presence")
    async def report_presence(room_id: str, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Report whether the user is actively viewing this room."""
        state = _room(room_id)
        active = bool(req.get("active", True))
        stack.note_presence(state, active)
        was_presence_paused = (
            state.last_decision is not None
            and state.last_decision.verdict is SelectionVerdict.SILENCE
            and "autonomous discussion paused" in (state.last_decision.reasoning or "")
        )
        if (
            active
            and state.policy.autonomous
            and was_presence_paused
            and not stack.turn_in_flight(room_id)
        ):
            orch = stack.orchestrator(state)
            last_seq = state.transcript[-1].seq if state.transcript else 0
            stack.drive(room_id, lambda: orch.resume(room_id, last_seq))
        return {"room_id": room_id, "active": active}

    @app.post("/api/rooms/{room_id}/retry")
    async def retry_turn(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Run a failed turn again without spending a fresh one."""
        state = _room(room_id)
        try:
            return (await stack.orchestrator(state).retry(room_id)).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/rooms/{room_id}/context")
    async def read_context(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """How full each seat's context is, and whether the conversation can still continue.

        Deliberately **not** folded into `GET /api/rooms/{room_id}`. That read is re-issued
        after every landed turn -- sending answers 202 and the head reconciles -- and a
        per-seat block on it would load N Core sessions on each reconciliation to answer a
        question a banner asks far less often.

        **`used_tokens` now carries a maximum, and it is a measured one.** It did not, and
        the reason it did not stands: `TokenBudget.max_tokens` is a spend ceiling
        (1,000,000 by default), so a proportion against it reads near nothing however full
        the context is, and a ring drawn against a ceiling that does not mean fullness is
        the plausible substituted value P6 forbids. What changed is that the other
        candidate stopped being unavailable. `uclone_x.llm.context_window` reads a locally
        served model's window from the daemon that loaded it and a hosted model's from the
        figure its provider publishes and enforces; where neither answers,
        `max_context_tokens` is `null` and the surface falls back to turns and says so.

        `saturation_threshold` is unaffected and is still the ceiling the Core enforces:
        a seat is saturated on *turns*, whatever its window. The two are different facts
        and both are sent, because a seat can be near either one first.

        `is_saturated` for the conversation is true when **any** seat is, because the
        conversation is what stops being able to continue: one full seat means one
        participant that cannot take another turn, and a room whose next speaker is that
        participant has stopped.

        **The seat read is translated like every other Core call in this module (#1212).**
        It was the one that was not. A seat nobody has spoken in is answered from its
        stored record, and `SessionStore.load` refuses a record that identifies a
        different session -- so a `SessionIdCollisionError` out of `active_turns` left
        this route with no `except` and reached the browser as a bare 500: no status that
        describes the conflict, no body, and none of the sentence that names the two ids
        and the one file they fold onto. A failure with no reason and no remedy is what
        P6 forbids.
        """
        state = _room(room_id)
        mgr = stack.session_manager()
        # Where this room's answers come from, which is what decides how a window can be
        # known at all. One read per request, not per seat: the daemon's answer covers
        # every model it has loaded, and asking once a seat would ask the same question
        # N times of the same endpoint.
        settings = mgr.get_settings()
        provider = str(settings.get("llm_provider") or "").strip().lower()
        llm_base_url = str(settings.get("llm_base_url") or "").strip()
        default_model = str(settings.get("llm_model") or "").strip()
        if provider == "ollama" and llm_base_url:
            await OLLAMA_CONTEXT_WINDOWS.refresh(llm_base_url)
        seats: list[dict[str, Any]] = []
        for participant in seated_agents(state):
            live = stack.live_agent(room_id, participant.session_id)
            try:
                active = mgr.active_turns(
                    agent_id=participant.id, session_id=participant.session_id, agent=live
                )
            except Exception as exc:  # the readout's one Core call (#1212)
                raise _http_error(exc) from exc
            # What this seat has actually spent, from the one place that knows.
            #
            # **Not from the transcript.** `RoomMessage.usage` is declared and nothing on
            # the orchestrated path populates it -- `TurnResult` carries no usage field for
            # the orchestrator to copy -- so a figure summed from the rows would be zero on
            # every room.
            # `BaseAgent` books each response against the budget manager instead, keyed by
            # session, and `_host()` above hands every seat that same manager.
            #
            # **`None` is not zero (P6).** The manager holds this process's bookings, so a
            # seat that has answered only in an earlier run has *no record*, which is a
            # different fact from having spent nothing. `get_budget` returns `None` for the
            # first and a zeroed budget for the second, and the two are sent as `null` and
            # `0`; `get_summary` cannot be used here because it flattens both to `0`.
            booked = mgr.budget_tracker.get_budget(participant.session_id)
            # The model this seat would answer on: its own when it is running with one,
            # and the configured default otherwise. Not read from the transcript -- a
            # `served_by` string is what served the *last* turn, and the window belongs to
            # the model that will serve the next one.
            seat_model = default_model
            if live is not None:
                configured = live.config.llm_config.model_name
                if configured:
                    seat_model = configured
            # A window is reported only where it was measured. `None` here is not a
            # missing feature to paper over with a typical value: it is the daemon not
            # having loaded this model yet, or a model nobody publishes a figure for, and
            # the head draws turns and says which (P6).
            window_tokens: int | None = None
            window_source: str | None = None
            if provider == "ollama":
                observed = OLLAMA_CONTEXT_WINDOWS.get(llm_base_url, seat_model)
                if observed is not None:
                    window_tokens, window_source = observed, "loaded"
            else:
                declared = published_context_window(provider, seat_model)
                if declared is not None:
                    window_tokens, window_source = declared, "published"
            # Active context tokens for the ring's denominator (current context window occupancy),
            # separate from cumulative spend across the session's lifetime.
            active_context_tokens: int | None = None
            if booked is not None:
                turn_history = mgr.budget_tracker.get_turn_history(participant.session_id)
                if turn_history:
                    active_context_tokens = (
                        turn_history[-1].input_tokens + turn_history[-1].output_tokens
                    )
                else:
                    active_context_tokens = booked.used_input_tokens + booked.used_output_tokens

            seats.append(
                {
                    "participant_id": participant.id,
                    "session_id": participant.session_id,
                    "active_turns": active,
                    "is_saturated": active >= SATURATION_TURNS_THRESHOLD,
                    "used_tokens": active_context_tokens,
                    "cumulative_tokens": (
                        None
                        if booked is None
                        else booked.used_input_tokens + booked.used_output_tokens
                    ),
                    # The denominator for that count, in the same unit, or `null`.
                    "max_context_tokens": window_tokens,
                    # Where the denominator came from, as a key rather than a sentence:
                    # the head writes the words a reader sees (P8).
                    "context_window_source": window_source,
                    # Which copy the figure came from, because the two can differ: a seat
                    # nobody has spoken in this process reads its persisted record, which
                    # is behind any turn the previous process did not write (P6).
                    "live": live is not None,
                }
            )
        return {
            "room_id": room_id,
            "seats": seats,
            "is_saturated": any(seat["is_saturated"] for seat in seats),
            "saturation_threshold": SATURATION_TURNS_THRESHOLD,
        }

    @app.post("/api/rooms/{room_id}/compact")
    async def compact_room(room_id: str, req: dict[str, Any] | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Shorten the context of every seat, or of the one named in `participant_id`.

        Room-level by default and per-seat by argument, because that is the order the two
        users ask in: the operator's question is *"this conversation is full, shorten it"*
        and does not mention seats, while naming one is a refinement that stays one argument
        away.

        Answers a result **per seat** and never a merged figure. Compaction is reported with
        the provenance of whichever producer wrote its ledger, and averaging four of those
        into one number invents an attribution no producer wrote.

        **`_refuse_during_turn` below is this route's early refusal, not its guard**
        (#1213). Unlike the two history routes it shares that helper with, this one awaits
        between asking and writing, so the answer it gets can be out of date by the time a
        later seat is rewritten. The guard is `BaseAgent.compact_session`'s own, run
        synchronously against each seat as it is compacted; it surfaces as a 409. See
        `_refuse_during_turn` for the whole asymmetry and what pins each half.
        """
        state = _room(room_id)
        _refuse_during_turn(stack, room_id, "Compacting")
        seats = seated_agents(state)
        raw = (req or {}).get("participant_id")
        if isinstance(raw, str) and raw:
            seats = tuple(p for p in seats if p.id == raw)
            if not seats:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{raw!r} is not a seated agent of this conversation, so there is "
                        f"no context of its own to shorten."
                    ),
                )
        results: list[dict[str, Any]] = []
        for participant in seats:
            try:
                agent = await stack.resolve_agent(state, participant)
                outcome = await agent.compact_session(participant.session_id, reason="ui_on_demand")
            except Exception as exc:
                raise _http_error(exc) from exc
            results.append(
                {
                    "participant_id": participant.id,
                    "session_id": outcome.session_id,
                    "reason": outcome.reason,
                    "ledger_source": outcome.ledger_source.value,
                    "tokens_before": outcome.tokens_before,
                    "tokens_after": outcome.tokens_after,
                    "saved_tokens": outcome.saved_tokens,
                    "compression_ratio_pct": outcome.compression_ratio_pct,
                    "messages_before": outcome.messages_before,
                    "messages_after": outcome.messages_after,
                    "superseded_ledger_count": outcome.superseded_ledger_count,
                    "active_turns": stack.session_manager().active_turns(
                        agent_id=participant.id,
                        session_id=outcome.session_id,
                        agent=agent,
                    ),
                    # Exactly as the Core stated it, `None` included (P6).
                    "provenance": (
                        outcome.provenance.model_dump(mode="json")
                        if outcome.provenance is not None
                        else None
                    ),
                }
            )
        return {"room_id": room_id, "results": results}

    @app.post("/api/rooms/{room_id}/history/truncate")
    async def truncate_history(room_id: str, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Rewind the conversation to a message, and cut every seat back to match.

        One transcript, several speakers: see `RoomService.truncate_transcript` for why a
        rewind cannot be a per-seat act, and `_reseat_after_history_change` for why each
        seat's own session is reset rather than index-mapped.
        """
        _room(room_id)
        _refuse_during_turn(stack, room_id, "Rewinding")
        raw = req.get("seq")
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise HTTPException(
                status_code=400,
                detail=(
                    "'seq' must be the number of the message to rewind to. A conversation "
                    "is rewound to a message it holds, not to a position in a list."
                ),
            )
        try:
            state = stack.service.truncate_transcript(room_id, raw)
        except Exception as exc:
            raise _http_error(exc) from exc
        return _history_answer(state, _reseat_after_history_change(stack, state))

    @app.delete("/api/rooms/{room_id}/history")
    async def clear_history(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Empty this conversation, keeping its id, its title and who is in it.

        Not `DELETE /api/rooms/{room_id}`, which removes the conversation, and not
        `POST /api/rooms`, which makes a different one alongside it. This is the one the
        operator means by *"start this over"*.
        """
        _room(room_id)
        _refuse_during_turn(stack, room_id, "Clearing")
        try:
            state = stack.service.clear_transcript(room_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return _history_answer(state, _reseat_after_history_change(stack, state))


def _history_answer(state: RoomState, kept_stale: tuple[str, ...]) -> dict[str, Any]:
    """The room, plus the seats whose own session did not reset with it.

    `participants_not_reset` is present on every answer, empty when everything reset. A key
    that appears only on failure is one a caller learns about by hitting the failure; a key
    that is always there and usually empty is one a caller can render from the start.
    """
    return {**state.model_dump(mode="json"), "participants_not_reset": list(kept_stale)}


def seated_agents(state: RoomState) -> tuple[Participant, ...]:
    """The room's agent participants that carry a session, in roster order.

    A participant with no `session_id` is skipped rather than defaulted: `RoomAgentResolver`
    refuses one, because an agent with no session of its own shares whichever session the
    host's default names, and a history control acting on that would cut back a session
    belonging to something else entirely.
    """
    return tuple(p for p in state.participants if p.kind is ParticipantKind.AGENT and p.session_id)


def _refuse_during_turn(stack: RoomStack, room_id: str, verb: str) -> None:
    """Refuse a history change while a turn is still running, and say why.

    The Core already refuses the equivalent on a chat session
    (`SessionMutationDuringTurnError`); a room's turn is driven behind an answered 202, so
    there is no request to hang the same refusal on and it has to be asked for here. A
    rewind under a running cascade is the same defect either way: the turn lands afterwards,
    carrying a `seq` derived from a transcript length it read before the cut, and writes a
    message into a conversation that no longer has the messages it was answering.

    **This check does not protect all three of its callers equally, and the difference is
    not visible from here (#1213).** Three routes ask it; what each gets differs:

    * `/history/truncate` and `DELETE /history` **are** protected by it, and only by it.
      Both run from this call to `RoomService.truncate_transcript` / `clear_transcript`
      and on through `_reseat_after_history_change` without a single `await`, so the whole
      handler is one uninterrupted pass of the event loop and no cascade can start between
      the question and the write. Nothing below the route re-asks: `RoomService` sees the
      store and knows nothing about `RoomStack._running`. That atomicity is therefore an
      invariant and not an implementation detail, and an `await` introduced anywhere in
      either handler silently retires it. Pinned by
      `tests/unit/test_ui_room_api.py::TestHistoryIsRefusedWhileSomebodyIsAnswering::
      test_the_two_history_routes_never_yield_between_the_question_and_the_write`.
    * `/compact` is **not** protected by it, and must not be read as if it were. That
      route awaits -- `resolve_agent`, then `compact_session` per seat -- so its answer
      here is a snapshot that a later seat's compaction can outlive. What covers it is
      `BaseAgent._refuse_session_mutation_during_turn`, which every `compact_session`
      runs synchronously on the seat it is about to rewrite, at the instant it rewrites
      it. That guard cannot go stale, and it is the one a caller meets: it raises
      `SessionMutationDuringTurnError`, which `_http_error` answers 409.

    So this call is compaction's *early, cheap* refusal -- it keeps an operator from
    starting a fan-out that the Core would decline seat by seat -- and it is the history
    routes' *only* one. Measured at `2d3f7207`: with an in-process connector the compact
    route did not in fact yield at all, even fanning out over two seats whose sessions
    were long enough to summarize, because `RoomAgentResolver.resolve` is an `async def`
    containing no `await` and the compactor's summarizer never suspends. The window opens
    when the summarizer does real I/O, which is every deployment that is not a test.

    Raises:
        HTTPException: 409, naming the control and the remedy.
    """
    if stack.turn_in_flight(room_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"{verb} this conversation is refused while somebody is still answering "
                f"in it: the turn would land after the change and write a reply to "
                f"messages that are no longer there. Stop the conversation first, then "
                f"try again."
            ),
        )


def _reseat_after_history_change(stack: RoomStack, state: RoomState) -> tuple[str, ...]:
    """Cut every seat's own session back to the transcript that now exists.

    **Reset and replay, not an index map.** A seat's session is not a slice of the
    transcript: it holds the spans the orchestrator built for it -- one user-role prompt per
    turn it was given, assembled from whatever it had not yet seen -- and its own replies.
    There is no message in it that corresponds to a `seq`, so any mapping from one to the
    other would be a reconstruction, and a reconstruction that drifted would leave a
    conversation whose reader and whose speakers remember different things: the silent
    failure this whole slice exists to avoid.

    So each seat is reset through the Core's single reset semantics, and the room's own
    record replays. `truncate_transcript` and `clear_transcript` have already cut
    `last_seen_seq` back, so `RoomOrchestrator._unseen_span` hands each speaker the surviving
    conversation again on its next turn, bounded by `RoomPolicy.max_span_messages` and
    announcing what that bound dropped. The cost is real and worth stating: a seat loses the
    compaction it had accumulated and re-reads the record. After a rewind that is the
    correct memory to have -- the surviving transcript is what happened.

    A seat with no live agent is reset through the same call, which deletes its stored
    record instead; the next resolve seeds it from config. Failures do not abort the loop:
    the transcript is already cut, and a seat left holding a stale session is recoverable
    where a half-applied reset reported as a failure is not.

    **Returns the seats that did not reset, and the caller says so in its answer.** A log
    line is not the answer: the caller reads 200 and a full `RoomState`, and a room where
    one seat still remembers the dropped turns is precisely the reader-and-speakers
    divergence above -- present, consequential, and invisible to the only party who could
    act on it. An absence states its cause rather than rendering as a silent success.
    """
    mgr = stack.session_manager()
    kept_stale: list[str] = []
    for participant in seated_agents(state):
        try:
            mgr.clear_session_history(
                agent_id=participant.id,
                session_id=participant.session_id,
                agent=stack.live_agent(state.room_id, participant.session_id),
            )
        except Exception:
            kept_stale.append(participant.id)
            logger.warning(
                "Room %s: seat %s kept its own session after a history change; it will "
                "answer from a conversation the record no longer holds until it is reset",
                state.room_id,
                participant.id,
                exc_info=True,
            )
    return tuple(kept_stale)


def _roll_back(service: RoomService, room_id: str, cause: Exception) -> None:
    """Undo a room that could not be seated, without losing why it could not.

    The delete had no failure path: when it raised, its own exception replaced the
    refusal the caller needed to read, and the half-seated room stayed in the list --
    both outcomes this route's docstring promised could not happen.
    """
    try:
        service.delete(room_id)
    except Exception:
        logger.error(
            "Room %s could not be seated (%s) and could not be removed either; it is "
            "left in the store and will appear in the listing",
            room_id,
            cause,
            exc_info=True,
        )


def _sole_human(state: RoomState) -> str:
    """The room's one human.

    Raises:
        HTTPException: 409 when nobody is seated. It used to answer `"user"`, which named
            a participant the caller never sent: typing then stamped an activity mark for
            somebody who does not exist, and sending was refused in a stranger's name.
    """
    for participant in state.participants:
        if participant.kind is ParticipantKind.HUMAN:
            return participant.id
    raise HTTPException(status_code=409, detail=_NO_HUMAN_REFUSAL)
