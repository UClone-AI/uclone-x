"""Reconstruct a turn's model calls, requests, responses, and tool results (#1490).

Provides the Core read model for inspecting a turn from the session log and context bodies.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.request_record import (
    RebuiltRequest,
    RequestRecordError,
    rebuild_requests,
)
from uclone_x.agent.session import SessionState
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.errors import UCloneXError

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


class TraceStep(BaseModel):
    """One step of an agent turn, pairing its request and response."""

    model_config = ConfigDict(extra="ignore")

    step: int
    request_status: Literal["ok", "unavailable"]
    request_reason: str | None = None
    verified: bool | None = None
    message_count: int | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_status: Literal["ok", "error", "unavailable"]
    response_reason: str | None = None
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
    verified: bool | None = None
    layers: dict[str, Any] | None = None
    response: ModelResponseRecord | None = None
    response_reason: str | None = None


#: The request reason when a step has no `REQUEST_CONTEXT` at all in the log. Distinct
#: from the pre-capture reason, which names a request that was recorded without its
#: snapshot: reporting one as the other would misstate what the log holds (P6).
NO_REQUEST_RECORDED = "no request is recorded for this step"
#: The response reason for a step with no `MODEL_RESPONSE`: logs before #1489.
NO_RESPONSE_RECORDED = "response recorded from #1489 on"
#: `TurnTrace.subagents_reason` when a tool result names a helper but cannot be read.
SUBAGENT_UNREADABLE = (
    "A tool result that names a helper could not be read, so a helper may be missing "
    "from this list."
)
#: Events written for a turn after its `TURN_END`: `TOOL_STEP_DROPPED` by the turn
#: itself (`base.py`, after the `TURN_END` append), `TURN_ROLLED_BACK` by the room
#: when the turn did not commit (`roll_back_turn`). Both carry the turn's `turn_index`.
_AFTER_END_TYPES = frozenset({"TOOL_STEP_DROPPED", "TURN_ROLLED_BACK"})
_NUDGE_TYPES = frozenset({"EVIDENCE_NUDGE", "EVIDENCE_NUDGE_DECLINED", "GROUNDING_NUDGE"})


def _extract_subagents(events: Sequence[Mapping[str, Any]]) -> tuple[list[str], str | None]:
    """Helper ids named by this turn's tool results, and why the list may be short.

    A `TOOL_RESULT.output` in the log is the canonical text of the result
    (`canonical_tool_text`): a delegation's is a JSON object with `subagent_id`. One
    that mentions `subagent_id` and does not parse is stated, not skipped.
    """
    seen: list[str] = []
    reason: str | None = None
    for ev in events:
        if ev.get("type") != "TOOL_RESULT":
            continue
        output = ev.get("output")
        parsed: Any = output
        if isinstance(output, str):
            if '"subagent_id"' not in output:
                continue
            try:
                parsed = json.loads(output)
            except ValueError:
                reason = SUBAGENT_UNREADABLE
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
    subagents, subagents_reason = _extract_subagents(turn_events)

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
        step_tool_results.setdefault(target, []).append(
            TraceToolResult(
                tool_call_id=call_id,
                name=str(ev.get("name", "")),
                status=ev.get("status"),
                outcome=ev.get("outcome"),
                output=ev.get("output"),
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
            }
        else:
            rebuild_err = step_errors.get(s)
            request_fields = {
                "request_status": "unavailable",
                "request_reason": (
                    rebuild_err.detail if rebuild_err is not None else NO_REQUEST_RECORDED
                ),
            }
        steps.append(
            TraceStep(
                step=s,
                **request_fields,
                response_status=(
                    "unavailable" if resp is None else "error" if resp.error is not None else "ok"
                ),
                response_reason=NO_RESPONSE_RECORDED if resp is None else None,
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
    _, _, turn_events, _ = _locate_turn(event_list, caller_turn_id)

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
    verified: bool | None = None
    layers_payload: dict[str, Any] | None = None
    rebuilt = rebuilt_by_step.get(step)
    if rebuilt is not None:
        request_dump = rebuilt.request.model_dump(mode="json")
        verified = rebuilt.verified
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
    else:
        request_reason = NO_REQUEST_RECORDED

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
        verified=verified,
        layers=layers_payload,
        response=_response_record(resp_event, step) if resp_event is not None else None,
        response_reason=None if resp_event is not None else NO_RESPONSE_RECORDED,
    )
