"""Reconstruct a turn's model calls, requests, responses, and tool results (#1490).

Provides the Core read model for inspecting a turn from the session log and context bodies.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.request_record import (
    EpochErrorCode,
    EpochRecordError,
    RebuiltRequest,
    RecordErrorCode,
    RequestRecordError,
    epoch_renderer,
    rebuild_requests,
)
from uclone_x.agent.session import SessionState
from uclone_x.core.context_state import ContextEpoch
from uclone_x.core.session_log import kept_result_text, stored_result_entry
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.tool_results import result_handle
from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage

__all__ = [
    "ModelResponseRecord",
    "StepDetail",
    "StepNotFoundError",
    "TraceStep",
    "TraceToolResult",
    "TurnNotLinkedError",
    "TurnTrace",
    "TurnTraceError",
    "trace_step",
    "trace_turn",
]


class TurnTraceError(UCloneXError):
    """Base error for turn trace inspection failures."""


class TurnNotLinkedError(TurnTraceError):
    """No `TURN_START` in the session log carries the turn's `caller_turn_id`."""


class StepNotFoundError(TurnTraceError):
    """The requested step was not found in the turn."""


class TraceToolResult(BaseModel):
    """The outcome of one tool call executed during a turn."""

    model_config = ConfigDict(extra="ignore")

    tool_call_id: str
    name: str
    status: str | None = None
    outcome: str | None = None
    output: Any = None
    #: The log names this result but the session no longer holds its text -- or the
    #: event is from before results were named by handle (#2013) -- so `output` is
    #: `None` and says nothing about what the call returned.
    output_unavailable: bool = False
    duration_ms: float | None = None
    at: str | None = None


class ModelResponseRecord(BaseModel):
    """The model response recorded for one step of a turn."""

    model_config = ConfigDict(extra="ignore")

    turn_index: int | None = None
    step: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    streamed: bool = False
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    finish_reason: str | None = None
    model_name: str | None = None
    usage: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


#: Why a step's conversation was not checked against the session log (#1903): the
#: session predates the context state, the request predates every epoch, or the step's
#: own epoch could not be rendered (`from_log_reason` then names what was missing). A
#: code, so the UI says it in the reader's language instead of quoting the reason.
FromLogCode = Literal["no_context_state", "no_epoch_for_request", "epoch_unreadable"]

#: Why a step's request is unavailable, as a stable code the UI words (#1907): the kind of
#: gap that stopped its rebuild (`RecordErrorCode`), or `not_recorded` for a step with no
#: `REQUEST_CONTEXT` in the log. `request_reason` stays the English detail.
RequestReasonCode = Literal[RecordErrorCode, "not_recorded"]
#: Why a step's response is unavailable (#1907): only `not_recorded`, a log before #1489.
ResponseReasonCode = Literal["not_recorded"]


class TraceStep(BaseModel):
    """One step of an agent turn, pairing its request and response."""

    model_config = ConfigDict(extra="ignore")

    step: int
    request_status: Literal["ok", "unavailable"]
    request_reason: str | None = None
    request_code: RequestReasonCode | None = Field(
        default=None,
        description="Why the request is unavailable, as a stable code the UI maps to its own "
        "sentence (#1907); `request_reason` stays the English detail.",
    )
    verified: bool | None = None
    message_count: int | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    from_log: bool | None = Field(
        default=None,
        description="Whether the step's conversation is what the session log and the "
        "context state render (#1848): a prefix of its own epoch's conversation, the "
        "last epoch opened at or before the step, rendered from the log alone. `None` "
        "when the request or that epoch could not be rebuilt.",
    )
    from_log_reason: str | None = None
    from_log_code: FromLogCode | None = Field(
        default=None,
        description="Why `from_log` is `None`, as a stable code the UI maps to its own "
        "sentence (#1903); `from_log_reason` stays the English detail.",
    )
    from_log_detail_code: EpochErrorCode | None = Field(
        default=None,
        description="For `epoch_unreadable`, what kind of gap stopped the epoch's rebuild, "
        "as a stable code (#1911), one of the four an epoch can raise (#1915); "
        "`from_log_reason` stays the English detail.",
    )
    response_status: Literal["ok", "error", "unavailable"]
    response_reason: str | None = None
    response_code: ResponseReasonCode | None = None
    response: ModelResponseRecord | None = None
    tool_results: list[TraceToolResult] = Field(default_factory=list[TraceToolResult])


class TurnTrace(BaseModel):
    """Full trace of one turn reconstructed from the session log."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    turn_index: int
    started_at: str | None = None
    ended_at: str | None = None
    steps: list[TraceStep] = Field(default_factory=list[TraceStep])
    nudges: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    rolled_back: bool = False
    dropped_tool_steps: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    turn_end: dict[str, Any] | None = None
    subagents: list[str] = Field(default_factory=list[str])
    subagents_reason: str | None = None


class StepDetail(BaseModel):
    """Detailed view of one step in a turn, including layered request and response."""

    model_config = ConfigDict(extra="ignore")

    step: int
    request: dict[str, Any] | None = None
    request_reason: str | None = None
    request_code: RequestReasonCode | None = None
    verified: bool | None = None
    layers: dict[str, Any] | None = None
    from_log: bool | None = Field(
        default=None,
        description="As `TraceStep.from_log`: whether the step's conversation is a prefix "
        "of its own epoch's conversation rendered from the session log (#1848).",
    )
    from_log_reason: str | None = None
    from_log_code: FromLogCode | None = None
    from_log_detail_code: EpochErrorCode | None = None
    response: ModelResponseRecord | None = None
    response_reason: str | None = None
    response_code: ResponseReasonCode | None = None


#: The request reason when a step has no `REQUEST_CONTEXT` at all in the log. Distinct
#: from the pre-capture reason, which names a request that was recorded without its
#: snapshot: reporting one as the other would misstate what the log holds (P6).
NO_REQUEST_RECORDED = "no request is recorded for this step"
#: The response reason for a step with no `MODEL_RESPONSE`: logs before #1489.
NO_RESPONSE_RECORDED = "response recorded from #1489 on"
#: `TurnTrace.subagents_reason` when a tool result cannot be read -- its text is no
#: longer held (#2013), or it names a helper and does not parse.
SUBAGENT_UNREADABLE = (
    "A tool result of this turn could not be read, so a helper may be missing from this list."
)
#: Events written for a turn after its `TURN_END`: `TOOL_STEP_DROPPED` by the turn
#: itself (`base.py`, after the `TURN_END` append), `TURN_ROLLED_BACK` by the room
#: when the turn did not commit (`roll_back_turn`). Both carry the turn's `turn_index`.
_AFTER_END_TYPES = frozenset({"TOOL_STEP_DROPPED", "TURN_ROLLED_BACK"})
_NUDGE_TYPES = frozenset({"EVIDENCE_NUDGE", "EVIDENCE_NUDGE_DECLINED", "GROUNDING_NUDGE"})


def logged_result_text(
    store: SessionStoreProtocol, state: SessionState, handle: object
) -> str | None:
    """The text of the tool result a `TOOL_RESULT` event names by `handle` (#2013).

    Read from the session's own log and bodies, as `tool_result_read` reads it: the
    entry the handle names, its body, which must still hash to the handle. `None` when
    the event names no handle or the session no longer holds that text.
    """
    if not isinstance(handle, str):
        return None
    entry = stored_result_entry(state.session_log, handle)
    if entry is None:
        return None
    try:
        body = store.load_context_body(state.session_id, entry.digest)
    except (OSError, UnicodeDecodeError):
        return None
    if body is None or result_handle(body) != handle:
        return None
    return kept_result_text(entry, body)


def _extract_subagents(
    events: Sequence[Mapping[str, Any]], read: Callable[[object], str | None]
) -> tuple[list[str], str | None]:
    """Helper ids named by this turn's tool results, and why the list may be short.

    A `TOOL_RESULT` names its result by handle (#2013), and `read` gives the canonical
    text (`canonical_tool_text`): a delegation's is a JSON object with `subagent_id`. One
    that cannot be read, or mentions `subagent_id` and does not parse, is stated, not
    skipped.
    """
    seen: list[str] = []
    reason: str | None = None
    for ev in events:
        if ev.get("type") != "TOOL_RESULT":
            continue
        text = read(ev.get("result_handle"))
        if text is None:
            reason = SUBAGENT_UNREADABLE  # a result the session no longer holds
            continue
        if '"subagent_id"' not in text:
            continue
        try:
            parsed: Any = json.loads(text)
        except ValueError:
            reason = SUBAGENT_UNREADABLE  # a result that names a helper and does not parse
            continue
        if not isinstance(parsed, dict):
            continue
        val = cast(dict[str, Any], parsed).get("subagent_id")
        if isinstance(val, str) and val and val not in seen:
            seen.append(val)
    return seen, reason


def _locate_turn(
    events: Sequence[Mapping[str, Any]],
    caller_turn_id: str,
) -> tuple[Mapping[str, Any], int, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """The turn's `TURN_START`, its index, its own events, and what was written after it.

    The turn runs from its `TURN_START` to the `TURN_END` with the same `turn_index`. A
    turn with no `TURN_END` (the process died mid-turn) ends at the next `TURN_START` or
    at the end of the log, never absorbing the turns after it. The events between its
    `TURN_END` and the next `TURN_START` are returned apart: the rollback and dropped
    tool step are read from them, by `turn_index`.
    """
    start_at: int | None = None
    for i, ev in enumerate(events):
        if ev.get("type") == "TURN_START" and ev.get("caller_turn_id") == caller_turn_id:
            start_at = i
            break
    if start_at is None:
        raise TurnNotLinkedError(f"no turn found with caller_turn_id {caller_turn_id!r}")
    turn_start_event = events[start_at]
    start_turn_index = int(turn_start_event["turn_index"])

    turn_events: list[Mapping[str, Any]] = [turn_start_event]
    after_end: list[Mapping[str, Any]] = []
    ended = False
    for ev in events[start_at + 1 :]:
        if ev.get("type") == "TURN_START":
            break
        if ended:
            after_end.append(ev)
            continue
        turn_events.append(ev)
        if ev.get("type") == "TURN_END" and ev.get("turn_index") == start_turn_index:
            ended = True
    return turn_start_event, start_turn_index, turn_events, after_end


def _response_record(ev: Mapping[str, Any], step: int) -> ModelResponseRecord:
    return ModelResponseRecord(
        turn_index=ev.get("turn_index"),
        step=step,
        started_at=ev.get("started_at"),
        ended_at=ev.get("ended_at"),
        streamed=bool(ev.get("streamed")),
        content=ev.get("content"),
        thinking=ev.get("thinking"),
        tool_calls=list(ev.get("tool_calls") or []),
        finish_reason=ev.get("finish_reason"),
        model_name=ev.get("model_name"),
        usage=ev.get("usage"),
        error=ev.get("error"),
    )


def _rebuild_turn_requests(
    store: SessionStoreProtocol,
    state: SessionState,
    events: Sequence[Mapping[str, Any]],
    wanted: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, RebuiltRequest], dict[int, RequestRecordError]]:
    """Rebuild the `wanted` `REQUEST_CONTEXT` events; every failure is kept by step."""
    if not wanted:
        return {}, {}
    wanted_ids = {id(ev) for ev in wanted}
    errors: dict[int, RequestRecordError] = {}

    def on_error(step: int, err: RequestRecordError) -> None:
        errors[step] = err

    rebuilt = rebuild_requests(
        store, state, events, select=lambda ev: id(ev) in wanted_ids, on_error=on_error
    )
    return {r.step: r for r in rebuilt}, errors


#: `from_log_reason` for a session recorded before the context state (#1443).
NO_CONTEXT_STATE = "no context state is recorded for this session"
#: `from_log_reason` for a request older than every epoch the context state records.
NO_EPOCH_FOR_REQUEST = "no epoch of the context state was opened at or before this request"


def _unchecked(
    code: FromLogCode, reason: str, detail_code: EpochErrorCode | None = None
) -> dict[str, Any]:
    """The `from_log` fields of a step that was not checked, with its code and reason.

    `detail_code` is the kind of gap behind `epoch_unreadable` (#1911); `None` otherwise.
    """
    return {
        "from_log": None,
        "from_log_reason": reason,
        "from_log_code": code,
        "from_log_detail_code": detail_code,
    }


class _LogCheck:
    """Checks a rebuilt request against its own epoch rendered from the session log.

    An epoch records the turn and step of the request that opened it, and the epochs
    are in the order they opened, so a request of turn `t`, step `s` belongs to the last
    epoch opened at or before `(t, s)`. Within an epoch the context only appends (§5.8,
    Rule 1), so each of its requests is a prefix of the conversation the epoch records.
    Comparing with the request's own epoch, not any epoch, is what catches a request
    that re-sent history an earlier epoch showed and a compaction replaced.

    Each epoch is rendered once, when a step first needs it. A missing or unreadable
    body is reported on the steps of that epoch, not raised, so the trace still shows
    the turn and the steps of other epochs are still checked (P6).
    """

    def __init__(self, store: SessionStoreProtocol, state: SessionState) -> None:
        self._epochs = state.context_epochs
        self._render = epoch_renderer(store, state)
        self._rendered: dict[int, list[ChatMessage] | EpochRecordError] = {}

    def _own_epoch(self, turn: int, step: int) -> ContextEpoch | None:
        own: ContextEpoch | None = None
        for epoch in self._epochs:
            if (epoch.turn, epoch.step) <= (turn, step):
                own = epoch
        return own

    def _conversation(self, epoch: ContextEpoch) -> list[ChatMessage] | EpochRecordError:
        if epoch.number not in self._rendered:
            try:
                self._rendered[epoch.number] = self._render(epoch)
            except EpochRecordError as err:
                self._rendered[epoch.number] = err
        return self._rendered[epoch.number]

    def fields(self, rebuilt: RebuiltRequest, turn: int) -> dict[str, Any]:
        """The `from_log` fields of a step of turn `turn` whose request was rebuilt."""
        if not self._epochs:
            return _unchecked("no_context_state", NO_CONTEXT_STATE)
        if rebuilt.layers is None:
            return {"from_log": None, "from_log_reason": None}
        epoch = self._own_epoch(turn, rebuilt.step)
        if epoch is None:
            return _unchecked("no_epoch_for_request", NO_EPOCH_FOR_REQUEST)
        rendered = self._conversation(epoch)
        if isinstance(rendered, EpochRecordError):
            return _unchecked("epoch_unreadable", rendered.detail, rendered.epoch_code)
        sent = list(rebuilt.layers.conversation)
        return {"from_log": rendered[: len(sent)] == sent}


def _step_number(ev: Mapping[str, Any]) -> int | None:
    raw = ev.get("step")
    if raw is None:
        return None
    return int(raw)


def trace_turn(
    store: SessionStoreProtocol,
    state: SessionState,
    events: Iterable[Mapping[str, Any]],
    *,
    caller_turn_id: str,
) -> TurnTrace:
    """Reconstruct the trace of one turn from the session log and context bodies.

    Locates the `TURN_START` whose `caller_turn_id` matches (`_locate_turn`), rebuilds the
    requests of that turn only, and pairs each step's request with the `MODEL_RESPONSE`
    of the same `step`. A tool result belongs to the step whose response asked for its
    `tool_call_id`; a log from before `MODEL_RESPONSE` (#1489) names no call on any step,
    and its results go to the step they follow in the log.

    A step whose request cannot be rebuilt is reported on that step with the reason;
    the other steps still rebuild if their chain allows (P6).

    Raises:
        TurnNotLinkedError: No `TURN_START` carries `caller_turn_id`.
    """
    event_list = list(events)
    turn_start_event, start_turn_index, turn_events, after_end = _locate_turn(
        event_list, caller_turn_id
    )

    turn_end_event = next(
        (
            ev
            for ev in turn_events
            if ev.get("type") == "TURN_END" and ev.get("turn_index") == start_turn_index
        ),
        None,
    )
    turn_end = {k: v for k, v in turn_end_event.items() if k != "type"} if turn_end_event else None
    own_after_end = [
        ev
        for ev in after_end
        if ev.get("type") in _AFTER_END_TYPES and ev.get("turn_index") == start_turn_index
    ]
    marks = [*turn_events, *own_after_end]
    rolled_back = any(ev.get("type") == "TURN_ROLLED_BACK" for ev in marks)
    dropped_tool_steps = [dict(ev) for ev in marks if ev.get("type") == "TOOL_STEP_DROPPED"]
    nudges = [dict(ev) for ev in turn_events if ev.get("type") in _NUDGE_TYPES]
    results: dict[str, str | None] = {}

    def read(handle: object) -> str | None:
        key = handle if isinstance(handle, str) else ""
        if key not in results:
            results[key] = logged_result_text(store, state, handle)
        return results[key]

    subagents, subagents_reason = _extract_subagents(turn_events, read)

    step_numbers: set[int] = set()
    step_responses: dict[int, ModelResponseRecord] = {}
    call_step: dict[str, int] = {}
    for ev in turn_events:
        step = _step_number(ev)
        if step is None:
            continue
        step_numbers.add(step)
        if ev.get("type") == "MODEL_RESPONSE":
            step_responses[step] = _response_record(ev, step)
            calls: list[Any] = list(ev.get("tool_calls") or [])
            for call in calls:
                call_id: Any = (
                    cast(Mapping[str, Any], call).get("id") if isinstance(call, Mapping) else None
                )
                if isinstance(call_id, str):
                    call_step[call_id] = step

    step_tool_results: dict[int, list[TraceToolResult]] = {}
    current_step: int | None = None
    for ev in turn_events:
        step = _step_number(ev)
        if step is not None:
            current_step = step
        if ev.get("type") != "TOOL_RESULT":
            continue
        call_id = str(ev.get("tool_call_id", ""))
        # By `tool_call_id` (§4.4.1); by position only when no response names the call.
        target = call_step.get(call_id, current_step if current_step is not None else 1)
        step_numbers.add(target)
        output = read(ev.get("result_handle"))
        step_tool_results.setdefault(target, []).append(
            TraceToolResult(
                tool_call_id=call_id,
                name=str(ev.get("name", "")),
                status=ev.get("status"),
                outcome=ev.get("outcome"),
                output=output,
                output_unavailable=output is None,
                duration_ms=ev.get("duration_ms"),
                at=ev.get("at"),
            )
        )

    rebuilt_by_step, step_errors = _rebuild_turn_requests(
        store,
        state,
        event_list,
        [ev for ev in turn_events if ev.get("type") == "REQUEST_CONTEXT"],
    )

    log_check = _LogCheck(store, state)
    steps: list[TraceStep] = []
    for s in sorted(step_numbers):
        rebuilt = rebuilt_by_step.get(s)
        resp = step_responses.get(s)
        if rebuilt is not None:
            request_fields: dict[str, Any] = {
                "request_status": "ok",
                "request_reason": None,
                "verified": rebuilt.verified,
                "message_count": len(rebuilt.request.messages),
                "model": rebuilt.request.model,
                "temperature": rebuilt.request.temperature,
                "max_tokens": rebuilt.request.max_tokens,
                **log_check.fields(rebuilt, start_turn_index),
            }
        else:
            rebuild_err = step_errors.get(s)
            request_fields = {
                "request_status": "unavailable",
                "request_reason": (
                    rebuild_err.detail if rebuild_err is not None else NO_REQUEST_RECORDED
                ),
                "request_code": rebuild_err.code if rebuild_err is not None else "not_recorded",
            }
        steps.append(
            TraceStep(
                step=s,
                **request_fields,
                response_status=(
                    "unavailable" if resp is None else "error" if resp.error is not None else "ok"
                ),
                response_reason=NO_RESPONSE_RECORDED if resp is None else None,
                response_code="not_recorded" if resp is None else None,
                response=resp,
                tool_results=step_tool_results.get(s, []),
            )
        )

    return TurnTrace(
        session_id=state.session_id,
        turn_index=start_turn_index,
        started_at=turn_start_event.get("at"),
        ended_at=turn_end_event.get("at") if turn_end_event else None,
        steps=steps,
        nudges=nudges,
        rolled_back=rolled_back,
        dropped_tool_steps=dropped_tool_steps,
        turn_end=turn_end,
        subagents=subagents,
        subagents_reason=subagents_reason,
    )


def trace_step(
    store: SessionStoreProtocol,
    state: SessionState,
    events: Iterable[Mapping[str, Any]],
    *,
    caller_turn_id: str,
    step: int,
) -> StepDetail:
    """One step's full request, its layers and its response, for step detail (#1490).

    Raises:
        TurnNotLinkedError: No `TURN_START` carries `caller_turn_id`.
        StepNotFoundError: The turn has no event for `step`.
    """
    event_list = list(events)
    _, turn_index, turn_events, _ = _locate_turn(event_list, caller_turn_id)

    if step not in {n for n in map(_step_number, turn_events) if n is not None}:
        raise StepNotFoundError(f"step {step} not found in turn {caller_turn_id}")

    target_rc = next(
        (
            ev
            for ev in turn_events
            if ev.get("type") == "REQUEST_CONTEXT" and _step_number(ev) == step
        ),
        None,
    )
    rebuilt_by_step, step_errors = _rebuild_turn_requests(
        store, state, event_list, [target_rc] if target_rc is not None else []
    )

    request_dump: dict[str, Any] | None = None
    request_reason: str | None = None
    request_code: RequestReasonCode | None = None
    verified: bool | None = None
    layers_payload: dict[str, Any] | None = None
    log_fields: dict[str, Any] = {}
    rebuilt = rebuilt_by_step.get(step)
    if rebuilt is not None:
        request_dump = rebuilt.request.model_dump(mode="json")
        verified = rebuilt.verified
        log_fields = _LogCheck(store, state).fields(rebuilt, turn_index)
        if rebuilt.layers is not None:
            layers_payload = {
                "identity": rebuilt.layers.identity,
                "slow_context": rebuilt.layers.slow_context,
                "turn_context": rebuilt.layers.turn_context,
                "tools_count": len(rebuilt.request.tools),
                "system_message": rebuilt.layers.system_message,
            }
    elif step in step_errors:
        request_reason = step_errors[step].detail
        request_code = step_errors[step].code
    else:
        request_reason = NO_REQUEST_RECORDED
        request_code = "not_recorded"

    resp_event = next(
        (
            ev
            for ev in turn_events
            if ev.get("type") == "MODEL_RESPONSE" and _step_number(ev) == step
        ),
        None,
    )
    return StepDetail(
        step=step,
        request=request_dump,
        request_reason=request_reason,
        request_code=request_code,
        verified=verified,
        layers=layers_payload,
        **log_fields,
        response=_response_record(resp_event, step) if resp_event is not None else None,
        response_reason=None if resp_event is not None else NO_RESPONSE_RECORDED,
        response_code=None if resp_event is not None else "not_recorded",
    )
