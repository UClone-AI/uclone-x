"""The dock's reads, scoped to one conversation and one seat (#1353, #1354, #1355, #1357).

Owner ruling, 2026-09-22: the dock describes the room on screen and the seat selected in
it. The workspace-wide and chat-scoped routes it read before (`/api/artifacts`,
`/api/agents`, `/api/knowledge-graph`) answer for the single-agent surface the centre
column retired in #1208, and a seat is invisible to all three: its agent is built and
cached by the room's resolver and never enters the chat manager's map. They stay for
compatibility; these are the room's own.

**Every read answers from the Core's record, and says when it cannot.** An empty list here
always means *recorded as empty* -- never that nothing happened: the room's file list in
particular never claims that no file was written, since a tool can write unseen (#1366).
Where the cause is that nothing was recorded -- a turn that raised before it could report
its tools, a seat not running in this process -- the field is `null` and a sentence says
why (P6). An unknown room or seat is a 404 that names
the roster, never an empty answer.

Kept out of `rooms.py`, which already holds the room's write surface and its stack.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import stat as stat_mode
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException

from uclone_x.agent.turn_trace import (
    StepDetail,
    StepNotFoundError,
    TurnNotLinkedError,
    trace_step,
    trace_turn,
)
from uclone_x.core.agent_home import AgentHomeError
from uclone_x.errors import (
    LogHeaderError,
    MemoryStoreUnreadableError,
    PathTraversalError,
    SeatKnowledgeUnreadableError,
    UnknownLogEventError,
)
from uclone_x.log import read_session_log
from uclone_x.memory.store import read_saved_facts
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFileRecord,
    RoomMessage,
    RoomState,
    RoomToolUse,
)
from uclone_x.room.service import participant_session_id
from uclone_x.room.turn_summary import TurnNotFoundError, summarize_turn
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.ui.knowledge import (
    knowledge_graph,
    remembered_statements,
    saved_fact_statements,
)
from uclone_x.ui.rooms import _http_error, seated_agents  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:  # pragma: no cover - import cycle; the app imports this module
    from uclone_x.agent.session import SessionState
    from uclone_x.ui.rooms import RoomStack

__all__ = ["register_room_dock_routes"]

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"})
_DOCUMENT_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".html"})

_NOT_RECORDED_RAISED = (
    "This turn ended before it could report its tools (it failed or was stopped), so which "
    "tools it used is not known."
)
_NOT_RECORDED_PARTIAL = (
    "This turn failed while its tools were running, so the tools listed for it may not be "
    "all it used."
)
#: What the room's file list covers, said with every read of it (#1366). The list is the
#: files a tool saved *and named*; a shell, an MCP server or a helper can write without
#: naming anything, so no read of this list ever says that nothing was written (P6).
FILES_SCOPE_NOTE = (
    "This list shows files the clones saved by name; a file written another way, such as "
    "by a shell command or a helper, may not appear here."
)

_NOT_RECORDED_LEGACY = (
    "This turn was recorded before the conversation kept a record of tool use, so which "
    "tools it used is not known."
)


def _turn_status(message: RoomMessage) -> str:
    if not message.completed:
        return "interrupted"
    if message.error is not None:
        return "failed"
    return "answered"


def _not_recorded_reason(message: RoomMessage, partial: bool) -> str | None:
    """Why a turn's tools are not known in full; `partial` when some calls were recorded."""
    if message.tools_recorded:
        return None
    if message.turn_id is None:
        return _NOT_RECORDED_LEGACY
    return _NOT_RECORDED_PARTIAL if partial else _NOT_RECORDED_RAISED


def _unsaved_turns(record: RoomFileRecord, turn_unlanded: bool) -> int:
    """Turns counted as started whose results never reached the store (#1366).

    The one turn this process is still running is not lost, and is subtracted. Negative
    means more turns landed than were counted as starting: the room began before turns
    were counted that way, so a lost turn from then would not show.
    """
    return record.turns_started - record.turns_landed - (1 if turn_unlanded else 0)


def _unsaved_gap(unsaved: int) -> str | None:
    # "Or still running elsewhere" (#1388 N5): the started/landed counts are in the room's
    # file, but which turn is running is known only to the process running it. A second
    # copy of the app serving the same room directory counts its turn as started, and this
    # process cannot tell that turn from a lost one, so it names both causes.
    if unsaved > 0:
        return (
            f"{unsaved} turn(s) started but stopped before their results were saved, or are "
            f"still running in another copy of the app"
        )
    if unsaved < 0:
        return "some turns were taken before this conversation counted turns as they started"
    return None


def _seat_turns(state: RoomState, participant_id: str) -> list[RoomMessage]:
    return [m for m in state.transcript if m.is_utterance and m.sender_id == participant_id]


def _use_payload(use: RoomToolUse, seq_by_turn: dict[str, int]) -> dict[str, Any]:
    # `seq` is `null` for a call whose turn is no longer in the transcript (a retried turn
    # keeps its failed row, so this is rare, but a lookup that guessed would misplace it).
    return {**use.model_dump(mode="json"), "seq": seq_by_turn.get(use.turn_id)}


def _roster_refusal(state: RoomState, participant_id: str) -> HTTPException:
    seats = [p.id for p in seated_agents(state)]
    listed = ", ".join(seats) if seats else "no agents are seated"
    return HTTPException(
        status_code=404,
        detail=(
            f"{participant_id!r} is not an agent seated in conversation {state.room_id!r} "
            f"(seated: {listed})."
        ),
    )


def _seat(state: RoomState, participant_id: str) -> Participant:
    seat = next((p for p in seated_agents(state) if p.id == participant_id), None)
    if seat is None:
        raise _roster_refusal(state, participant_id)
    return seat


def _artifact_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _DOCUMENT_SUFFIXES:
        return "document"
    return "file"


def _file_on_disk(workspace: Path, path: str) -> dict[str, Any]:
    """`exists` and `size_bytes` for one recorded path, and why when they are unknown.

    `exists` is True or False only when a `stat` answered that question. A path outside
    the workspace is never read, and a `stat` that failed for any reason but absence
    (a permission, an I/O error) did not answer it, so both are `null` with the cause
    (P6) -- "does not exist" is a claim neither of them made.
    """
    try:
        resolved = PathValidator().resolve_safe_path(Path(path), workspace)
    except PathTraversalError:
        return {
            "exists": None,
            "size_bytes": None,
            "exists_reason": "This path is outside the workspace, so it was not checked.",
        }
    except ValueError:  # a path the OS cannot name, such as one holding a null byte
        return {
            "exists": None,
            "size_bytes": None,
            "exists_reason": "This path cannot be checked on this computer.",
        }
    try:
        found = resolved.stat()
    except FileNotFoundError:
        return {"exists": False, "size_bytes": None, "exists_reason": None}
    except OSError as exc:
        return {
            "exists": None,
            "size_bytes": None,
            "exists_reason": f"Could not check this file: {exc.strerror or type(exc).__name__}.",
        }
    size = found.st_size if stat_mode.S_ISREG(found.st_mode) else None
    return {"exists": True, "size_bytes": size, "exists_reason": None}


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _file_record_gaps(
    file_record: RoomFileRecord, legacy_agent_turns: int, unsaved: int
) -> list[str]:
    """Each known reason the room's file list may be missing a write, in plain words.

    Read from the room's monotonic record, not from the rows that remain: a clear or a
    rewind removes the rows that showed a gap, and the files stay on disk (#1366).
    """
    gaps: list[str] = []
    if not file_record.kept_since_creation:
        gaps.append("this conversation began before files written by tools were recorded")
    elif legacy_agent_turns:
        gaps.append(f"{legacy_agent_turns} turn(s) were saved without a record of their tools")
    if file_record.unrecorded_turns:
        gaps.append(f"{file_record.unrecorded_turns} turn(s) ended before reporting their tools")
    unsaved_gap = _unsaved_gap(unsaved)
    if unsaved_gap is not None:
        gaps.append(unsaved_gap)
    if file_record.unattributed_writes:
        gaps.append(
            f"{file_record.unattributed_writes} tool call(s) could have written files "
            f"without naming them"
        )
    if file_record.clears:
        gaps.append("its history was cleared")
    if file_record.rewinds:
        gaps.append("its history was rewound")
    return gaps


def _tool_call_gaps(record: RoomFileRecord, legacy_agent_turns: int) -> list[str]:
    """Each known reason a turn in the graph may show fewer tool calls than it made (#1388 N3).

    Apart from `history_gaps`, which names turn rows that may be missing: every row can be
    present while a turn's calls are not. A helper's own calls are never listed and are not
    a gap here: that is the graph's scope, not a lapse in its record, and the seat history's
    `tools_note` and the head's topology tab say so with every read.
    """
    gaps: list[str] = []
    if legacy_agent_turns:
        gaps.append(f"{legacy_agent_turns} turn(s) were saved without a record of their tools")
    if record.unrecorded_turns:
        gaps.append(f"{record.unrecorded_turns} turn(s) ended before reporting their tools")
    return gaps


def _legacy_agent_turns(state: RoomState) -> int:
    """Agent rows with no turn id: saved by a build that did not record tools.

    Only agent rows: a human's row has no turn id either, and wrote nothing.
    """
    humans = {p.id for p in state.participants if p.kind is ParticipantKind.HUMAN}
    return sum(
        1
        for m in state.transcript
        if m.is_utterance and m.turn_id is None and m.sender_id not in humans
    )


def _no_turns_reason(state: RoomState, name: str, unsaved: int, in_progress: bool) -> str:
    """Why a seat has no turns, claiming "not yet" only when none can have been removed
    or lost, and none is in progress."""
    record = state.file_record
    if not record.kept_since_creation:
        return (
            f"{name} has no turns in this conversation now. This conversation is older "
            f"than the record of clearing and rewinding its history, so whether {name} "
            f"took earlier turns is not known."
        )
    removed: list[str] = []
    if record.clears:
        removed.append("cleared")
    if record.rewinds:
        removed.append("rewound")
    causes: list[str] = []
    if removed:
        causes.append(f"its history was {' and '.join(removed)}")
    lost = _unsaved_gap(unsaved)
    if lost is not None:
        causes.append(lost)
    if causes:
        return (
            f"{name} has no turns in this conversation now, but {_join(causes)}, so any "
            f"earlier turns by {name} are not shown and whether there were any is not known."
        )
    if in_progress:
        return (
            f"{name} has no saved turn in this conversation yet. A turn is in progress now "
            f"and appears here when it is saved."
        )
    return f"{name} has not taken a turn in this conversation yet."


def _saved_memory(seat: Participant) -> dict[str, Any]:
    """The seat's saved memory facts, or why they cannot be listed (#1401).

    Read from the memory the seat's `record_memory_fact` writes: the store the room's
    resolver composes the seat with is `memory_for(participant.id)`, one document per agent
    id, shared with that clone in every other conversation. Read-only -- a GET never moves a
    damaged document aside, and never creates the agent's home.
    """
    name = seat.display_name
    try:
        facts = read_saved_facts(seat.id)
    except AgentHomeError:
        # An id that cannot name an agent home has no memory to read: `memory_for` refuses
        # it too, so no fact can have been saved under it.
        facts = None
    except MemoryStoreUnreadableError as exc:
        # The path and the parser's words go to the log; the person reading the dock can act
        # on neither.
        logger.warning(
            "Saved memory of seat %r could not be read: %s",
            seat.id,
            getattr(exc, "cause", None) or exc,
        )
        return {
            "saved_facts": None,
            "saved_facts_reason": f"{name}'s saved facts could not be read, so they cannot be shown.",
        }
    listed = saved_fact_statements(facts or [], seat.session_id or "")
    reason = None if listed else f"No saved facts are listed for {name}."
    return {"saved_facts": listed, "saved_facts_reason": reason}


#: Why a turn cannot be traced (the turn-inspection design, §4.4.3). The code is for
#: tests and the UI's logic; the message is what the UI prints. None of them carries a
#: path or an exception's text: the routes are not gated on developer mode (§8 Q1).
_TRACE_REASONS: dict[str, dict[str, str]] = {
    code: {"code": code, "message": message}
    for code, message in (
        ("not_an_agent_turn", "This turn was spoken by a person, not an agent."),
        (
            "turn_not_linked",
            "This turn was recorded before turns were linked to their session (#1489), "
            "so its model calls cannot be found.",
        ),
        (
            "turn_not_saved",
            "This turn's record was not saved, so its model calls cannot be shown.",
        ),
        ("session_not_found", "No saved session was found for this seat."),
        (
            "session_unreadable",
            "This seat's saved session could not be read, so the turn cannot be traced.",
        ),
        ("log_missing", "No session log was found for this seat."),
        (
            "log_unreadable",
            "This seat's session log could not be read, so the turn cannot be traced.",
        ),
        (
            "trace_failed",
            "This turn's record could not be read back, so its model calls cannot be shown.",
        ),
    )
}


def _log_failure_kind(exc: BaseException) -> str:
    """A stable name for why `read_session_log` failed, safe to send to the client."""
    if isinstance(exc, UnknownLogEventError):
        return "unknown_event_type"
    if isinstance(exc, LogHeaderError):
        return "malformed_log"
    if isinstance(exc, UnicodeDecodeError):
        return "not_text"
    if isinstance(exc, OSError):
        return "read_failed"
    return "unexpected"


def _spoken_by_agent(state: RoomState, message: RoomMessage) -> bool:
    """Whether `message` is an agent's turn, from the room's record rather than its roster.

    A seat still in the room answers by its kind. A seat that left is no longer in
    `participants` (`remove_participant` drops it), and its rows then answer for
    themselves: only an agent's turn carries a turn id, a speaker decision or a
    provenance -- a person's row and a membership row carry none of them.
    """
    if not message.is_utterance:
        return False
    participant = next((p for p in state.participants if p.id == message.sender_id), None)
    if participant is not None:
        return participant.kind == ParticipantKind.AGENT
    linked = message.turn_id is not None or message.decision is not None
    return linked or message.provenance is not None


def _turn_not_found(room_id: str, seq: int) -> HTTPException:
    """The 404 for a seq the room's transcript does not hold."""
    return HTTPException(
        status_code=404,
        detail={
            "code": "turn_not_found",
            "message": f"Turn {seq} was not found in conversation {room_id}.",
        },
    )


def _turn_not_linked_reason(message: RoomMessage) -> dict[str, str]:
    """Why a turn with an id has no record: its row's save failed, or the log lacks it."""
    unsaved = message.persist_error is not None
    return _TRACE_REASONS["turn_not_saved" if unsaved else "turn_not_linked"]


def register_room_dock_routes(app: FastAPI, stack: RoomStack) -> None:
    """Mount the dock's room-scoped reads under `/api/rooms/{room_id}`."""
    service = stack.service

    def _room(room_id: str) -> RoomState:
        try:
            return service.get(room_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/api/rooms/{room_id}/turns/{seq}")
    async def read_turn_summary(room_id: str, seq: int) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Summary of what one turn did for general user inspection (#1491).

        Returns 200 with TurnSummary model dump.
        Raises 404 with turn_not_found if the turn seq does not exist in the room.
        """
        state = _room(room_id)
        try:
            summary = summarize_turn(state, seq)
        except TurnNotFoundError as exc:
            raise _turn_not_found(room_id, seq) from exc
        return summary.model_dump(mode="json")

    @app.get("/api/rooms/{room_id}/seats/{participant_id}/history")
    async def read_seat_history(room_id: str, participant_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """One seat's turns in this conversation, and the tools each one used (#1353).

        From the room's record, not the seat's session: the record is written with the
        turn, survives a restart and a compaction, and is there for a seat that is not
        running. A turn whose tools were not recorded carries `tools: null` and the reason,
        never `[]` -- a turn that raised has no account of its tools, and "none" would be
        a claim nobody made.
        """
        state = _room(room_id)
        seat = _seat(state, participant_id)
        turns = _seat_turns(state, seat.id)
        seq_by_turn = {m.turn_id: m.seq for m in state.transcript if m.turn_id is not None}
        uses = [u for u in state.tool_uses if u.participant_id == seat.id]
        by_turn: dict[str, list[dict[str, Any]]] = {}
        for use in uses:
            by_turn.setdefault(use.turn_id, []).append(_use_payload(use, seq_by_turn))
        unsaved = _unsaved_turns(state.file_record, stack.turn_unlanded(state.room_id))
        in_progress = stack.turn_in_flight(state.room_id) or stack.turn_unlanded(state.room_id)
        turn_rows: list[dict[str, Any]] = []
        for message in turns:
            recorded = message.tools_recorded and message.turn_id is not None
            turn_rows.append(
                {
                    "seq": message.seq,
                    "turn_id": message.turn_id,
                    "created_at": message.created_at,
                    "status": _turn_status(message),
                    "error": message.error,
                    "content_preview": message.content[:280],
                    "tools": by_turn.get(message.turn_id or "", []) if recorded else None,
                    "tools_not_recorded_reason": _not_recorded_reason(
                        message, bool(by_turn.get(message.turn_id or ""))
                    ),
                }
            )
        return {
            "room_id": state.room_id,
            "participant_id": seat.id,
            "display_name": seat.display_name,
            "session_id": seat.session_id,
            "live": stack.live_agent(state.room_id, seat.session_id) is not None,
            "turns": turn_rows,
            "tool_uses": [_use_payload(u, seq_by_turn) for u in uses],
            # What `tools` covers, said with every read: a seat's own calls, not what a
            # helper it started did, nor every file a call touched. An empty list is "none
            # listed", never "it used no tools" as a fact about the world (#1366).
            "tools_note": (
                f"This lists the tool calls {seat.display_name}'s saved turns reported. A "
                f"helper {seat.display_name} started does its work with tools of its own, "
                f"which are not listed separately."
            ),
            # Turns anywhere in the room whose results were lost before they were saved.
            # Room-wide, because the count does not say whose turn it was: any of them
            # could have been this seat's, so its list may be missing one (#1366).
            "unsaved_turns": max(unsaved, 0),
            "unsaved_note": (
                None
                if unsaved <= 0
                else f"{unsaved} turn(s) in this conversation started but stopped before "
                f"their results were saved, or are still running in another copy of the "
                f"app, so a turn by {seat.display_name} may be missing here."
            ),
            "reason": (
                None
                if turn_rows
                else _no_turns_reason(state, seat.display_name, unsaved, in_progress)
            ),
        }

    def _traced_message(room_id: str, seq: int) -> tuple[RoomState, RoomMessage]:
        state = _room(room_id)
        message = next((m for m in state.transcript if m.seq == seq), None)
        if message is None:
            raise _turn_not_found(room_id, seq)
        return state, message

    def _unlinked_reason(state: RoomState, message: RoomMessage) -> dict[str, str] | None:
        """Why this row cannot be traced before any record is read, or None."""
        if not _spoken_by_agent(state, message):
            return _TRACE_REASONS["not_an_agent_turn"]
        if message.turn_id is None:
            return _TRACE_REASONS["turn_not_linked"]
        return None

    def _read_trace_inputs(
        message: RoomMessage, session_id: str
    ) -> tuple[Any, SessionState, list[dict[str, Any]]] | dict[str, str]:
        """The seat's store, saved session and log events, or the reason one is missing.

        Blocking file I/O: callers run it in a thread. A row whose save failed
        (`persist_error`) and whose record is absent reports `turn_not_saved`, not the
        absence: the failed save is why it is absent.
        """
        unsaved = message.persist_error is not None
        store = stack.session_manager().core_store
        try:
            session_state = store.load(session_id)
        except Exception:
            logger.warning("Saved session %s could not be read", session_id, exc_info=True)
            return _TRACE_REASONS["session_unreadable"]
        if session_state is None:
            return _TRACE_REASONS["turn_not_saved" if unsaved else "session_not_found"]
        log_path = store.event_log_path(session_id)
        if log_path is None or not log_path.exists():
            return _TRACE_REASONS["turn_not_saved" if unsaved else "log_missing"]
        try:
            events = list(read_session_log(log_path))
        except Exception as exc:
            # The exception text names the file and quotes the line: it goes to the
            # server log, and the reader gets a stable code for what kind of failure.
            logger.warning("Session log of %s could not be read", session_id, exc_info=True)
            return {**_TRACE_REASONS["log_unreadable"], "detail": _log_failure_kind(exc)}
        return store, session_state, events

    async def _off_loop(
        work: Any, session_id: str
    ) -> tuple[Literal["read", "reason", "no_step"], dict[str, Any]]:
        """Run `work` in a thread; an unexpected failure is a stated reason, not a 500.

        `work` answers `("read", payload)`, `("reason", reason)` or `("no_step", {})`.
        """
        try:
            return await asyncio.to_thread(work)
        except Exception:
            logger.exception("Tracing a turn of session %s failed", session_id)
            return "reason", _TRACE_REASONS["trace_failed"]

    @app.get("/api/rooms/{room_id}/turns/{seq}/trace")
    async def read_turn_trace(room_id: str, seq: int) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Full trace of one turn's model calls, requests, and tool results (#1490).

        Reconstructs the trace from the session log and context bodies.
        Returns 200 with the trace, or with `trace: null` and a reason (§4.4.3 of
        the turn-inspection design). Raises 404 if the turn seq does not exist.
        """
        state, message = _traced_message(room_id, seq)
        participant_id = message.sender_id
        session_id = participant_session_id(room_id, participant_id)
        head: dict[str, Any] = {
            "room_id": room_id,
            "seq": seq,
            "turn_id": message.turn_id,
            "participant_id": participant_id,
            "session_id": session_id,
        }
        reason = _unlinked_reason(state, message)
        if reason is not None:
            return {**head, "trace": None, "reason": reason}
        turn_id = message.turn_id
        assert turn_id is not None  # _unlinked_reason refused a row without one

        def _load_and_trace() -> tuple[Literal["read", "reason"], dict[str, Any]]:
            inputs = _read_trace_inputs(message, session_id)
            if isinstance(inputs, dict):
                return "reason", inputs
            store, session_state, events = inputs
            try:
                trace = trace_turn(store, session_state, events, caller_turn_id=turn_id)
            except TurnNotLinkedError:
                return "reason", _turn_not_linked_reason(message)
            return "read", trace.model_dump(mode="json")

        kind, outcome = await _off_loop(_load_and_trace, session_id)
        if kind == "read":
            return {**head, "trace": outcome}
        return {**head, "trace": None, "reason": outcome}

    @app.get("/api/rooms/{room_id}/turns/{seq}/trace/steps/{step}")
    async def read_turn_step_detail(room_id: str, seq: int, step: int) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """One step's full request and response for developer inspection (#1490).

        Returns 200 with {step, request, request_reason, verified, layers, response,
        response_reason, reason}. `reason` is null when the step was read, and otherwise
        carries the same code the trace route gives for the turn, with the other fields
        null. Raises 404 turn_not_found for an unknown seq and step_not_found for a step
        the turn does not have.
        """
        state, message = _traced_message(room_id, seq)
        session_id = participant_session_id(room_id, message.sender_id)
        empty = StepDetail(step=step).model_dump(mode="json")
        reason = _unlinked_reason(state, message)
        if reason is not None:
            return {**empty, "reason": reason}
        turn_id = message.turn_id
        assert turn_id is not None  # _unlinked_reason refused a row without one

        def _load_and_trace_step() -> tuple[Literal["read", "reason", "no_step"], dict[str, Any]]:
            inputs = _read_trace_inputs(message, session_id)
            if isinstance(inputs, dict):
                return "reason", inputs
            store, session_state, events = inputs
            try:
                detail = trace_step(store, session_state, events, caller_turn_id=turn_id, step=step)
            except TurnNotLinkedError:
                return "reason", _turn_not_linked_reason(message)
            except StepNotFoundError:
                return "no_step", {}
            return "read", detail.model_dump(mode="json")

        kind, outcome = await _off_loop(_load_and_trace_step, session_id)
        if kind == "no_step":
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "step_not_found",
                    "message": f"Step {step} was not found for turn {seq}.",
                },
            )
        if kind == "read":
            return {**outcome, "reason": None}
        return {**empty, "reason": outcome}

    @app.get("/api/rooms/{room_id}/artifacts")
    async def read_room_artifacts(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The files this conversation's seats wrote through tools, newest write first (#1354).

        Exactly the paths the room recorded -- not the workspace's `artifacts/`, its
        `docs/` or its root Markdown, which `GET /api/artifacts` still lists for callers
        that want the workspace. One entry per path however often it was rewritten.
        `exists` is read from the disk now: a recorded file can since have been deleted,
        and listing it as present would be the substituted value P6 forbids. It is `null`,
        with `exists_reason`, when the disk did not answer (see `_file_on_disk`).

        **It never says that nothing was written** (#1366). A tool can write without the
        room seeing it -- a shell, an MCP server, a helper a seat started -- so an empty
        list is only ever "no files are listed", with `scope_note` saying what the list
        covers and `record_gaps` naming each known reason it may be missing a write. An
        empty `record_gaps` means no *known* gap, not a complete list.
        """
        state = _room(room_id)
        workspace = stack.session_manager().workspace_dir
        entries: dict[str, dict[str, Any]] = {}
        for written in state.written_files:
            entry: dict[str, Any] | None = entries.get(written.path)
            if entry is None:
                entry = {
                    "id": f"art_{hashlib.sha256(written.path.encode('utf-8')).hexdigest()[:12]}",
                    "path": written.path,
                    "name": Path(written.path).name,
                    "title": Path(written.path).name,
                    "type": _artifact_type(written.path),
                    "writers": [],
                    "write_count": 0,
                }
                entries[written.path] = entry
            entry["write_count"] += 1
            if written.participant_id not in entry["writers"]:
                entry["writers"].append(written.participant_id)
            # The latest write names the entry: who last changed it, with what, when.
            entry.update(
                {
                    "participant_id": written.participant_id,
                    "tool_name": written.tool_name,
                    "turn_id": written.turn_id,
                    "tool_call_id": written.tool_call_id,
                    "written_at": written.written_at,
                }
            )
        for entry in entries.values():
            entry.update(_file_on_disk(workspace, entry["path"]))
        artifacts = sorted(entries.values(), key=lambda e: str(e["written_at"]), reverse=True)
        record = state.file_record
        unattributed = record.unattributed_writes
        unrecorded = record.unrecorded_turns
        # The known gaps are read from the room's monotonic record, not from the rows that
        # remain: a clear or a rewind removes the rows that showed a gap while the files
        # stay on disk (#1366).
        unlanded = stack.turn_unlanded(state.room_id)
        unsaved = _unsaved_turns(record, unlanded)
        gaps = _file_record_gaps(record, _legacy_agent_turns(state), unsaved)
        # A cascade between turns, or a turn in progress -- a retry runs its turn inside
        # the request, not as a cascade, and was invisible to `turn_in_flight` alone.
        running = stack.turn_in_flight(state.room_id) or unlanded
        causes = list(gaps)
        if running:
            # Its writes reach the record when it finishes, not before.
            causes.append("a turn is still running and its files are listed when it finishes")
        reason: str | None = None
        if not artifacts:
            # Never "no file was written": that cannot be known (see the docstring).
            reason = "No files are listed yet. " + FILES_SCOPE_NOTE
            if causes:
                reason += " It may also be missing files because " + _join(causes) + "."
        return {
            "room_id": state.room_id,
            "artifacts": artifacts,
            "total": len(artifacts),
            "unattributed_writes": unattributed,
            "unattributed_note": (
                None
                if unattributed == 0
                else f"{unattributed} tool call(s) in this conversation could have written "
                f"files without naming them (a shell command or a helper, for example), so "
                f"files they touched may not be listed here."
            ),
            "unrecorded_turns": unrecorded,
            "unrecorded_note": (
                None
                if unrecorded == 0
                # "May not all be": a turn that failed partway has some files listed (#1388).
                else f"{unrecorded} turn(s) in this conversation ended (raised or were "
                f"stopped) before reporting their tools, so the files they wrote may not "
                f"all be listed here."
            ),
            "unsaved_turns": max(unsaved, 0),
            # No `record_complete`: an empty `record_gaps` is no known gap, never a proof
            # that the list holds every file written (#1366).
            "record_gaps": gaps,
            "scope_note": FILES_SCOPE_NOTE,
            "turn_running": running,
            "reason": reason,
        }

    @app.get("/api/rooms/{room_id}/topology")
    async def read_room_topology(room_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The conversation as a graph: its seats, their turns, their tools, their sub-agents (#1355).

        `summary.tool_calls` counts the calls listed, never all calls made: `history_gaps`
        names why turns may be missing, `tool_call_gaps` why a turn's calls may be (#1388).

        Every seated agent is a node whether or not it has spoken -- an idle seat is a
        fact about the room, and a graph that drops it reads as "no agents". A seat that
        has since left keeps its node, marked `left`, because its turns are still there.
        """
        state = _room(room_id)
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seated = {p.id: p for p in seated_agents(state)}
        roster = {p.id: p for p in state.participants}
        turns = [
            m
            for m in state.transcript
            if m.is_utterance
            and (m.sender_id in seated or roster.get(m.sender_id) is None)
            and not (m.sender_id in roster and roster[m.sender_id].kind is ParticipantKind.HUMAN)
        ]
        spoken = {m.sender_id for m in turns}
        for participant in seated.values():
            nodes.append(
                {
                    "id": f"seat:{participant.id}",
                    "kind": "seat",
                    "participant_id": participant.id,
                    "display_name": participant.display_name,
                    "session_id": participant.session_id,
                    "status": "answered" if participant.id in spoken else "idle",
                    "turn_count": sum(1 for m in turns if m.sender_id == participant.id),
                    "live": stack.live_agent(state.room_id, participant.session_id) is not None,
                }
            )
        for former in sorted(spoken - set(seated)):
            nodes.append(
                {
                    "id": f"seat:{former}",
                    "kind": "seat",
                    "participant_id": former,
                    "display_name": former,
                    "session_id": None,
                    "status": "left",
                    "turn_count": sum(1 for m in turns if m.sender_id == former),
                    "live": False,
                }
            )
        uses_by_turn: dict[str, list[RoomToolUse]] = {}
        for use in state.tool_uses:
            uses_by_turn.setdefault(use.turn_id, []).append(use)
        previous: str | None = None
        tool_calls = 0
        subagents: set[str] = set()
        for message in turns:
            turn_node = f"turn:{message.turn_id or f'seq-{message.seq}'}"
            nodes.append(
                {
                    "id": turn_node,
                    "kind": "turn",
                    "participant_id": message.sender_id,
                    "seq": message.seq,
                    "turn_id": message.turn_id,
                    "status": _turn_status(message),
                    "tools_recorded": message.tools_recorded,
                }
            )
            edges.append(
                {
                    "id": f"edge:{message.sender_id}:{turn_node}",
                    "source": f"seat:{message.sender_id}",
                    "target": turn_node,
                    "kind": "took_turn",
                }
            )
            if previous is not None:
                edges.append(
                    {
                        "id": f"edge:{previous}:{turn_node}",
                        "source": previous,
                        "target": turn_node,
                        "kind": "followed_by",
                    }
                )
            previous = turn_node
            for index, use in enumerate(uses_by_turn.get(message.turn_id or "", [])):
                tool_calls += 1
                tool_node = f"tool:{message.turn_id}:{index}"
                nodes.append(
                    {
                        "id": tool_node,
                        "kind": "tool",
                        "participant_id": use.participant_id,
                        "tool_name": use.tool_name,
                        "tool_call_id": use.tool_call_id,
                        "status": use.status,
                        "turn_id": use.turn_id,
                    }
                )
                edges.append(
                    {
                        "id": f"edge:{turn_node}:{tool_node}",
                        "source": turn_node,
                        "target": tool_node,
                        "kind": "called",
                    }
                )
                if use.subagent_id is not None:
                    child = f"subagent:{use.subagent_id}"
                    if use.subagent_id not in subagents:
                        subagents.add(use.subagent_id)
                        nodes.append(
                            {
                                "id": child,
                                "kind": "subagent",
                                "subagent_id": use.subagent_id,
                                "parent_participant_id": use.participant_id,
                            }
                        )
                    edges.append(
                        {
                            "id": f"edge:seat:{use.participant_id}:{child}",
                            "source": f"seat:{use.participant_id}",
                            "target": child,
                            "kind": "spawned",
                        }
                    )
        # What the graph cannot show: its turns and tool calls are the rows that remain, so
        # a count of zero is not "none ever" when rows were removed or never saved (#1366).
        # `history_gaps` and `history_complete` speak for turn rows only; a turn that is
        # present can still be missing tool calls, named in `tool_call_gaps` (#1388 N3).
        record = state.file_record
        history_gaps: list[str] = []
        if not record.kept_since_creation:
            history_gaps.append("this conversation began before its history was fully recorded")
        if record.clears:
            history_gaps.append("its history was cleared")
        if record.rewinds:
            history_gaps.append("its history was rewound")
        unsaved_gap = _unsaved_gap(_unsaved_turns(record, stack.turn_unlanded(state.room_id)))
        if unsaved_gap is not None:
            history_gaps.append(unsaved_gap)
        tool_call_gaps = _tool_call_gaps(record, _legacy_agent_turns(state))
        return {
            "room_id": state.room_id,
            "nodes": nodes,
            "edges": edges,
            # True when no turn row is known to be missing. Not "every tool call is shown":
            # `tool_call_gaps` names why a present turn may be missing calls (#1388 N3).
            "history_complete": not history_gaps,
            "history_gaps": history_gaps,
            "tool_call_gaps": tool_call_gaps,
            "summary": {
                "seats": len(seated),
                "turns": len(turns),
                "tool_calls": tool_calls,
                "subagents": len(subagents),
            },
            "reason": None if seated else "No agent is seated in this conversation.",
        }

    @app.get("/api/rooms/{room_id}/knowledge")
    async def read_seat_knowledge(room_id: str, agent_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """What one seat has learned, from that seat's own engine (#1357, G4).

        Never the manager's shared engine: a seat is composed with an engine of its own
        (P7), and the shared one describes nothing it learned.

        A running seat is read from its agent. A seat that is not running is read from its
        saved knowledge record (#1367), loaded into a detached
        engine -- the route does not build the seat to find out, because building it is
        what a turn does. Each answer that is not a memory says which it is:

        * `not_recorded`: not running and nothing saved -- not an empty memory;
        * `unreadable`: a saved record is there and could not be read;
        * `no_ontology`: running without a knowledge store.

        `remembers` is the same content as `triples`, as plain statements for a reader who
        is not a programmer; the head writes the words around them.

        `saved_facts` is the other half of what the clone remembers, and on every answer
        whatever the knowledge record's status (#1401): the facts it saved to memory with
        `record_memory_fact`, from its own memory -- never another clone's. `null` with
        `saved_facts_reason` when they could not be read; an empty list with a reason saying
        none are listed.
        """
        state = _room(room_id)
        if not agent_id:
            seats = ", ".join(p.id for p in seated_agents(state)) or "no agents are seated"
            raise HTTPException(
                status_code=400,
                detail=f"Name the seat to read with agent_id (seated: {seats}).",
            )
        seat = _seat(state, agent_id)
        base: dict[str, Any] = {
            "room_id": state.room_id,
            "participant_id": seat.id,
            "session_id": seat.session_id,
            # Read whatever the knowledge record says: a seat's saved facts are in its
            # memory, not in its knowledge engine (#1401).
            **_saved_memory(seat),
        }
        empty: dict[str, Any] = {
            "triples": None,
            "nodes": None,
            "edges": None,
            "summary": None,
            "remembers": None,
        }
        live = stack.live_agent(state.room_id, seat.session_id)
        if live is not None and live.ontology is None:
            return {
                **base,
                **empty,
                "status": "no_ontology",
                "reason": f"{seat.display_name} is running without a knowledge store.",
            }
        if live is not None:
            engine = live.ontology
        else:
            try:
                engine = stack.knowledge.read(seat.session_id, seat.ontology_namespace)
            except SeatKnowledgeUnreadableError:
                # Plain words only: the store logged where the record is and why it could
                # not be read, and a reader of the dock can act on neither. The read itself
                # changes nothing.
                name = seat.display_name
                return {
                    **base,
                    **empty,
                    "status": "unreadable",
                    "reason": (
                        f"{name}'s knowledge record for this conversation could not be read, "
                        f"so it cannot be shown."
                    ),
                }
            if engine is None:
                return {
                    **base,
                    **empty,
                    "status": "not_recorded",
                    "reason": (
                        f"{seat.display_name} has no knowledge record in this conversation."
                    ),
                }
        graph = knowledge_graph(engine)
        remembers = remembered_statements(graph["triples"])
        return {
            **base,
            **graph,
            "remembers": remembers,
            "status": "ok",
            "reason": None,
        }
