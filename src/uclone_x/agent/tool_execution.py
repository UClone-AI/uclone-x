"""Running one tool call: hooks, approval, the tool range, the call itself (#1736).

`BaseAgent._execute_tools` decides which calls a step runs and in what order; this module
runs each one. A call passes the `PRE_TOOL_USE` hooks and, when one asks, a person's
approval; it is refused if its name is outside the agent's tool range, unknown, or
withheld by the persona flags; otherwise the tool runs, an `update_plan` result is
applied to the session plan, and the `POST_TOOL_USE` hooks see the result. Every path
returns the tool message the model reads and the record the turn keeps.

The executor owns no state. What it reads from the agent it serves -- config, context,
hooks, the bus, the tool catalog, the persona flags and the plan -- comes through an
`ExecutionScope` of callables read on every call, so a value changed after construction
is the value a call sees.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from uclone_x.agent.hooks import HookAction, HookContext, HookEvent, HookRunner
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    PlanState,
    PlanStep,
    ToolExecutionRecord,
)
from uclone_x.agent.tool_invoker import ToolInvoker, unadvertised_tool_message
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.tool_results import canonical_tool_text
from uclone_x.engine.event_bus import AgentEvent, EventType
from uclone_x.engine.protocols import EventBusProtocol, PublisherHandleProtocol
from uclone_x.errors import PathTraversalError
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.tools.base import (
    tool_approval_timeout_note,
    tool_approval_unavailable_note,
    tool_call_needs_approval,
    tool_call_writes_files,
    tool_opens_story,
    tool_spawns_subagents,
    tool_writes_files,
)
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.outcome import (
    EMPTY_RESULT_NOTE,
    ToolOutcome,
    classify_payload_shape,
    classify_tool_outcome,
)

__all__ = [
    "ExecutionScope",
    "PlanSession",
    "ToolCallExecutor",
    "tool_outcome_of",
]

logger = logging.getLogger(__name__)


def _modified_arguments(modified_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The call arguments a hook's `modified_payload` sets, read as execution reads them.

    `{"arguments": {...}}` sets them; anything else sets none (#1504).
    """
    if not modified_payload:
        return None
    arguments: object = modified_payload.get("arguments")
    if isinstance(arguments, Mapping):
        return cast(dict[str, Any], unwrap_immutable(cast(Mapping[str, Any], arguments)))
    return None


def tool_outcome_of(record: ToolExecutionRecord) -> str:
    """The errored / empty / productive classification, for the log.

    `ToolExecutionRecord` is what the turn keeps; `classify_tool_outcome` reads a
    `ToolResult`. Rather than thread the result through, the record's own fields are
    mapped the same way -- one place, and the mapping is asserted against the classifier
    in `tests/unit/test_turn_step_logging.py` so the two cannot drift.
    """
    if record.status != ToolResultStatus.SUCCESS:
        return ToolOutcome.ERRORED.value
    return classify_payload_shape(record.output).value


class PlanSession(Protocol):
    """The part of a live session an `update_plan` result writes: its plan."""

    plan: PlanState | None


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """The agent state a `ToolCallExecutor` reads, as callables read on every call."""

    agent_id: Callable[[], str]
    context: Callable[[], AgentContext]
    config: Callable[[], AgentConfig]
    tool_invoker: Callable[[], ToolInvoker]
    hook_runner: Callable[[], HookRunner]
    #: Whether anyone answers an approval request (`BaseAgent(approvals_answered=...)`).
    approvals_answered: Callable[[], bool]
    bus: Callable[[], EventBusProtocol | None]
    publisher: Callable[[], PublisherHandleProtocol | None]
    #: The persona-flag refusal for a resolved tool, or `None` (`BaseAgent._capability_refusal`).
    capability_refusal: Callable[[object], str | None]
    current_plan: Callable[[], PlanState | None]
    #: `BaseAgent.create_plan(title, steps)`.
    create_plan: Callable[[str, Sequence[str | PlanStep | Mapping[str, Any]]], PlanState]
    #: `BaseAgent.update_step_status(index, completed, verification)`.
    update_step_status: Callable[[int, bool, str | None], PlanState]
    live_session: Callable[[str], PlanSession]
    publish_plan_update: Callable[[PlanState], None]


class ToolCallExecutor:
    """Runs one tool call for the agent it serves and returns its message and record.

    The accessors below carry the names the agent's own attributes have, so the method
    reads exactly as it did on `BaseAgent`; each one reads the scope, never a copy.
    """

    def __init__(self, scope: ExecutionScope) -> None:
        self._scope = scope

    @property
    def agent_id(self) -> str:
        return self._scope.agent_id()

    @property
    def _context(self) -> AgentContext:
        return self._scope.context()

    @property
    def _config(self) -> AgentConfig:
        return self._scope.config()

    @property
    def _tool_invoker(self) -> ToolInvoker:
        return self._scope.tool_invoker()

    @property
    def _hook_runner(self) -> HookRunner:
        return self._scope.hook_runner()

    @property
    def _approvals_answered(self) -> bool:
        return self._scope.approvals_answered()

    @property
    def _bus(self) -> EventBusProtocol | None:
        return self._scope.bus()

    @property
    def _publisher(self) -> PublisherHandleProtocol | None:
        return self._scope.publisher()

    @property
    def current_plan(self) -> PlanState | None:
        return self._scope.current_plan()

    def _capability_refusal(self, tool: object) -> str | None:
        return self._scope.capability_refusal(tool)

    def create_plan(
        self, *, title: str, steps: Sequence[str | PlanStep | Mapping[str, Any]]
    ) -> PlanState:
        return self._scope.create_plan(title, steps)

    def update_step_status(
        self, *, index: int, completed: bool, verification: str | None
    ) -> PlanState:
        return self._scope.update_step_status(index, completed, verification)

    def _live_session(self, session_id: str) -> PlanSession:
        return self._scope.live_session(session_id)

    def _publish_plan_update(self, plan: PlanState) -> None:
        self._scope.publish_plan_update(plan)

    def _refused_tool_call(
        self,
        tc: ToolCallRequest,
        err_msg: str,
        duration_ms: float,
    ) -> tuple[ChatMessage, ToolExecutionRecord]:
        """The synthesized tool message and record for a call the pre-tool path refused.

        One function rather than two identical literals, because the duplication carried
        a defect. `ToolExecutionRecord.arguments` is `ImmutableJsonMapping`, and
        `tc.arguments` is frozen *recursively*, so a shallow `dict()` copy of it left
        nested `MappingProxyType`s in place. Those do not survive the field's
        `JsonValue` validation — `AfterValidator(freeze_mapping)` runs after it, so the
        field rejects the shallow copy rather than re-freezing it — and the refusal path
        raised `ValidationError` instead of returning the refusal it exists to return
        (#665).
        """
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content=err_msg,
            name=tc.name,
            tool_call_id=tc.id,
        )
        rec = ToolExecutionRecord(
            tool_name=tc.name,
            arguments=cast(dict[str, Any], unwrap_immutable(tc.arguments)),
            output=None,
            status=ToolResultStatus.ERROR,
            error=err_msg,
            duration_ms=duration_ms,
            tool_call_id=tc.id,
        )
        return msg, rec

    async def execute_single_tool(
        self,
        tc: ToolCallRequest,
        tool_ctx: ToolContext,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        advertised: frozenset[str] | None = None,
    ) -> tuple[ChatMessage, ToolExecutionRecord]:
        """Execute a single tool call within the provided ToolContext, intercepted by hooks."""
        t_start = asyncio.get_running_loop().time()

        # F12: a call runs only if the request that produced it declared its name. Checked
        # before the hooks, so a call that will not run never asks a person to approve it.
        # A name this agent would refuse anyway -- not allowed, not registered, withheld by
        # the persona flags -- falls through to that refusal, which keeps its own words.
        # R3 (#2190): a held catalog tool that binding missed is bound on this call and
        # runs, through the hooks below like any other call.
        if (
            advertised is not None
            and tc.name not in advertised
            and self._tool_invoker.would_run_if_advertised(tc.name)
            and not self._tool_invoker.bind_on_call(tc.name, tool_ctx.session_id)
        ):
            return self._refused_tool_call(
                tc,
                unadvertised_tool_message(tc.name),
                (asyncio.get_running_loop().time() - t_start) * 1000.0,
            )

        # 1. Execute PRE_TOOL_USE hook
        pre_payload: dict[str, Any] = {
            "tool_name": tc.name,
            "tool_call_id": tc.id,
            "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
        }
        # What the tool declares about itself, so a hook can decide from that and not from
        # the tool's name (#1463). Absent for a name this agent cannot resolve.
        pre_tool = self._tool_invoker.resolve(tc.name)
        if pre_tool is not None:
            pre_payload["writes_files"] = tool_writes_files(pre_tool)
            pre_payload["spawns_subagents"] = tool_spawns_subagents(pre_tool)
            # A call that runs only once a person approves it: the hook runner answers
            # `ASK` for it whatever the hooks say (#1557).
            pre_payload["needs_approval"] = tool_call_needs_approval(
                pre_tool, pre_payload["arguments"]
            )
        pre_ctx = HookContext(
            agent_id=self.agent_id,
            session_id=self._context.session_id,
            trace_id=self._context.trace_id,
            event_type=HookEvent.PRE_TOOL_USE,
            payload=pre_payload,
        )
        pre_decision = await self._hook_runner.run_hooks(HookEvent.PRE_TOOL_USE, pre_ctx)
        # Whether a person answered this call's approval request with yes. Only that sets
        # `ToolContext.approved_by_person`; a hook's ALLOW does not (#1557).
        approved_by_person = False

        if pre_decision.action == HookAction.ASK and not self._approvals_answered:
            # Nobody here answers an approval request (the desktop app), so waiting out the
            # timeout would only delay the same refusal. Refused before anything is asked,
            # fail-closed: the model still cannot approve its own call.
            unavailable_note = (
                tool_approval_unavailable_note(pre_tool)
                if pre_payload.get("needs_approval") is True
                else None
            )
            return self._refused_tool_call(
                tc,
                unavailable_note
                or "This call needs a person's approval, and this app cannot ask for it "
                "during a conversation, so it did not run.",
                (asyncio.get_running_loop().time() - t_start) * 1000.0,
            )

        if pre_decision.action == HookAction.ASK:
            request_id = f"appr_{uuid.uuid4().hex[:8]}"

            sub = None
            if self._bus is not None:
                sub = self._bus.subscribe({f"session.{self._context.session_id}"})

            # The call as the hooks left it: an `ASK` that follows a hook's `MODIFY` carries
            # the rewrite, and the person is asked about -- and approves -- that call (#1584).
            hook_arguments = _modified_arguments(pre_decision.modified_payload)
            approval_arguments = (
                hook_arguments
                if hook_arguments is not None
                else cast(dict[str, Any], unwrap_immutable(tc.arguments))
            )
            if self._bus is not None:
                req_evt = AgentEvent(
                    type=EventType.TOOL_APPROVAL_REQUEST,
                    topic=f"session.{self._context.session_id}",
                    sender_id=self.agent_id,
                    payload={
                        "request_id": request_id,
                        "tool_call_id": tc.id,
                        "tool_name": tc.name,
                        "arguments": approval_arguments,
                        "reason": pre_decision.reason,
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                    },
                    trace_id=self._context.trace_id,
                )
                if self._publisher is not None:
                    await self._publisher.publish(req_evt)
                else:
                    await self._bus.publish(req_evt)

            try:
                if sub is None:
                    raise TimeoutError()

                async def wait_for_response() -> Mapping[str, Any]:
                    while True:
                        evt = await sub.get()
                        if evt.type == EventType.TOOL_APPROVAL_RESPONSE:
                            payload = evt.payload
                            rid = payload.get("request_id")
                            tid = payload.get("tool_call_id")
                            if rid == request_id or tid == tc.id:
                                return payload

                payload = await asyncio.wait_for(
                    wait_for_response(),
                    timeout=getattr(self._config, "approval_timeout_seconds", 30.0),
                )
                from uclone_x.agent.hooks.models import ApprovalDecision

                # `payload` is `AgentEvent.payload` — frozen recursively, so
                # `payload.get("modified_arguments")` yields a `MappingProxyType`, while
                # `ApprovalDecision.modified_arguments` is `dict[str, Any]` under
                # `strict=True` and rejects it with `dict_type` (#665, #673). This
                # `ApprovalDecision(...)` sits in a `try` whose only handler is
                # `except TimeoutError`, so a MODIFY approval response would take the
                # turn down rather than degrade. Latent — no in-repo publisher sets the
                # key — and invisible to the fitness sweep, which matches
                # `dict(<expr>.<field>)` and not an aliased `.get()`.
                modified_arguments = payload.get("modified_arguments")
                decision = ApprovalDecision(
                    action=HookAction(payload.get("action", "allow")),
                    reason=payload.get("reason"),
                    modified_arguments=cast(dict[str, Any], unwrap_immutable(modified_arguments))
                    if modified_arguments is not None
                    else None,
                    decided_by=payload.get("decided_by"),
                )

                # Override pre_decision with human decision
                from uclone_x.agent.hooks.models import HookDecision

                # The person's own rewrite wins; otherwise what they approved is the call as
                # the hooks left it, not the model's original (#1584). Tested against `None`,
                # never for truth: an empty rewrite, `{}`, is shown and so is what runs.
                approved_arguments = (
                    decision.modified_arguments
                    if decision.modified_arguments is not None
                    else hook_arguments
                    if decision.action in (HookAction.ALLOW, HookAction.MODIFY)
                    else None
                )
                pre_decision = HookDecision(
                    action=HookAction.MODIFY
                    if approved_arguments is not None and decision.action == HookAction.ALLOW
                    else decision.action,
                    reason=decision.reason,
                    modified_payload={"arguments": approved_arguments}
                    if approved_arguments is not None
                    else None,
                )
                approved_by_person = decision.action in (HookAction.ALLOW, HookAction.MODIFY)
            except TimeoutError:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                # A call that needed a person says so in the tool's own words. The desktop
                # app does not reach here: it refuses before asking (`approvals_answered`).
                timeout_note = (
                    tool_approval_timeout_note(pre_tool)
                    if pre_payload.get("needs_approval") is True
                    else None
                )
                return self._refused_tool_call(
                    tc,
                    timeout_note or "Approval request timed out (denied fail-closed)",
                    duration_ms,
                )
            finally:
                if sub is not None:
                    sub.close()

        if pre_decision.action == HookAction.BLOCK:
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            block_reason = pre_decision.reason or "Blocked by hook"
            return self._refused_tool_call(
                tc, f"Tool execution blocked by hook: {block_reason}", duration_ms
            )

        effective_tc = tc
        if pre_decision.action == HookAction.MODIFY and pre_decision.modified_payload is not None:
            mod_payload = pre_decision.modified_payload
            if "arguments" in mod_payload and isinstance(mod_payload["arguments"], dict):
                effective_tc = tc.model_copy(update={"arguments": mod_payload["arguments"]})
            elif "tool_name" not in mod_payload and "arguments" not in mod_payload:
                effective_tc = tc.model_copy(update={"arguments": mod_payload})

        unwrapped_args: dict[str, Any] = cast(
            dict[str, Any], unwrap_immutable(effective_tc.arguments)
        )

        if stream_callback is not None:
            try:
                res_status = stream_callback(
                    "status",
                    {"status": "calling_tool", "detail": f"Running tool: {effective_tc.name}..."},
                )
                if asyncio.iscoroutine(res_status):
                    await res_status
                res_tool = stream_callback(
                    "tool_call",
                    {
                        "tool": effective_tc.name,
                        "args": unwrapped_args,
                        "output": None,
                        "status": "running",
                    },
                )
                if asyncio.iscoroutine(res_tool):
                    await res_tool
            except Exception:
                logger.debug("Error in stream_callback during tool start", exc_info=True)

        # The allowlist is checked here, not only at advertise time. `execute_turn` shows the
        # model the permitted names and this used to be a bare registry lookup, so a call the
        # model produced for a name it was never shown ran anyway -- and small models produce
        # exactly that, which is why `text_tool_calls` exists. Advertising is a hint; this is
        # the enforcement.
        #
        # Checked *before* the lookup so the two answers stay separate: a permitted-but-absent
        # tool and a present-but-forbidden one must not be reported with the same words.
        # `execute_tool_call` raises `PermissionError` and `KeyError` for the same distinction,
        # and a turn should not collapse what a direct call keeps apart. This check is the
        # *only* refusal: a `ScopedToolRegistry`'s lookup finds a registered tool outside its
        # scope (#908) -- the scope governs what `list_tools` advertises, not what `get` finds.
        allowed_names = self._config.allowed_tools
        if not self._tool_invoker.knows_tool(effective_tc.name):
            # R4 (#2190): a name no tool has. Answered before the range check, in plain
            # words with the closest tools this agent may use, so an invented name is
            # neither told the agent's permission list nor left without a way forward.
            did_you_mean = self._tool_invoker.unknown_tool_result(effective_tc.name, unwrapped_args)
            return self._refused_tool_call(
                effective_tc, did_you_mean, (asyncio.get_running_loop().time() - t_start) * 1000.0
            )
        if not self._tool_invoker.in_tool_range(effective_tc.name):
            err_msg = (
                f"Tool '{effective_tc.name}' is not in agent "
                f"'{self.agent_id}' allowed_tools {tuple(allowed_names)!r}"
            )
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            return (
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=err_msg,
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                ),
                ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=err_msg,
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                ),
            )

        tool_inst = self._tool_invoker.resolve(effective_tc.name)
        if tool_inst is None:
            # Registered, but only as another agent's own instance (`ToolInvoker.resolve`).
            err_msg = self._tool_invoker.unknown_tool_result(effective_tc.name, unwrapped_args)
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=err_msg,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = ToolExecutionRecord(
                tool_name=effective_tc.name,
                arguments=unwrapped_args,
                output=None,
                status=ToolResultStatus.ERROR,
                error=err_msg,
                duration_ms=duration_ms,
                tool_call_id=effective_tc.id,
            )
        elif (refusal := self._capability_refusal(tool_inst)) is not None:
            # The persona flags, refused on the same path as `allowed_tools` above and in
            # the same shape (#1167): an error record, and the tool is never executed. After
            # the lookup rather than before it, because what the flags refuse is a kind of
            # tool, and only the instance says which kind it is.
            duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=refusal,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = ToolExecutionRecord(
                tool_name=effective_tc.name,
                arguments=unwrapped_args,
                output=None,
                status=ToolResultStatus.ERROR,
                error=refusal,
                duration_ms=duration_ms,
                tool_call_id=effective_tc.id,
            )
        else:
            # The tool's declarations, read once: a call that raises partway reports
            # them too, since failing does not undo a write (#1366). A call to an action
            # the tool declares read-only did not write (#1584).
            declared_writes = tool_call_writes_files(tool_inst, unwrapped_args)
            declared_spawns = tool_spawns_subagents(tool_inst)
            call_ctx = (
                tool_ctx.model_copy(update={"approved_by_person": True})
                if approved_by_person
                else tool_ctx
            )
            try:
                res = await tool_inst.execute(unwrapped_args, call_ctx)
                duration_ms = (
                    res.execution_time_ms
                    if res.execution_time_ms > 0
                    else (asyncio.get_running_loop().time() - t_start) * 1000.0
                )

                # Update session plan natively if the tool was the plan tool
                if effective_tc.name == "update_plan" and res.success:
                    if isinstance(res.output, dict):
                        try:
                            action = str(res.output.get("action"))
                            if action == "create":
                                title = str(res.output.get("title", ""))
                                steps_val = res.output.get("steps")
                                steps_list = steps_val if isinstance(steps_val, list) else []
                                self.create_plan(
                                    title=title,
                                    steps=steps_list,  # type: ignore[reportArgumentType]
                                )
                            elif action == "update":
                                if not self.current_plan:
                                    raise RuntimeError("No active plan exists to update")
                                steps_val = res.output.get("steps")
                                if isinstance(steps_val, list):
                                    for step in steps_val:
                                        if isinstance(step, dict):
                                            idx = step.get("index")
                                            if isinstance(idx, int):
                                                verif = step.get("verification")
                                                self.update_step_status(
                                                    index=idx,
                                                    completed=bool(step.get("completed", False)),
                                                    verification=str(verif)
                                                    if verif is not None
                                                    else None,
                                                )
                                status_val = res.output.get("status")
                                if isinstance(status_val, str) and status_val:
                                    new_plan = self.current_plan.model_copy(
                                        update={"status": status_val}
                                    )
                                    self._live_session(self._context.session_id).plan = new_plan
                                    self._publish_plan_update(new_plan)

                            # Provide the new plan back to the tool result so LLM sees success explicitly
                            if self.current_plan:
                                plan_dump = {
                                    "plan_id": self.current_plan.plan_id,
                                    "title": self.current_plan.title,
                                    "status": self.current_plan.status,
                                    "steps": [s.model_dump() for s in self.current_plan.steps],
                                }
                            else:
                                plan_dump = {}
                            res = res.model_copy(
                                update={
                                    "output": {
                                        "message": f"Plan {action}d successfully",
                                        "plan": plan_dump,
                                    }
                                }
                            )
                        except Exception as e:
                            logger.error("Failed to update session plan from tool output: %s", e)
                            res = res.model_copy(
                                update={"success": False, "error": str(e), "output": None}
                            )

                # Canonical text, not `str()`: a dict result was a Python repr -- not JSON,
                # quoted with `'`, and ordered by insertion (#1422).
                content = canonical_tool_text(res.output) if res.success else str(res.error)
                # A call that succeeded and found nothing arrives structurally
                # indistinguishable from one that answered the question -- both are
                # `success=True`, and `{"total_matches": 0, "matches": []}` reads as an
                # absence. Measured: a model missed with a regex, concluded "the line does
                # not appear in the file", and stopped. The number was there (#698). The
                # note says what the result is; it does not say what to do next, which is
                # the model's judgement and not something the runtime can make for it.
                if classify_tool_outcome(res) is ToolOutcome.EMPTY:
                    content = f"{content}\n{EMPTY_RESULT_NOTE}"
                if (
                    not res.success
                    and res.error
                    and ("Path traversal" in res.error or "PathTraversalError" in res.error)
                ):
                    logger.warning(
                        "Path traversal violation during tool '%s' execution: %s",
                        effective_tc.name,
                        res.error,
                        extra={
                            "agent_id": self.agent_id,
                            "session_id": self._context.session_id,
                            "tool_name": effective_tc.name,
                            "error": str(res.error),
                        },
                    )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=content,
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                    # A picture travels beside the text, never inside it (#2107): the
                    # text is what the event log and the stream show.
                    images=res.images if res.success else (),
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=res.output,
                    status=ToolResultStatus.SUCCESS if res.success else ToolResultStatus.ERROR,
                    error=res.error,
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # The tool's declarations, carried to whoever reads the turn: a room
                    # records a written file only from a tool that says it writes (#1354).
                    writes_files=declared_writes,
                    spawns_subagents=tool_spawns_subagents(tool_inst),
                    # Only a call that succeeded may move the conversation's story (#1555).
                    opens_story=res.success and tool_opens_story(tool_inst),
                    # What the tool says it produced; read through `produced_paths` (#2085).
                    artifacts=res.artifacts if res.success else (),
                )
            except PathTraversalError as exc:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                logger.warning(
                    "Path traversal violation during tool '%s' execution: %s",
                    effective_tc.name,
                    exc,
                    extra={
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                        "tool_name": effective_tc.name,
                        "error": str(exc),
                    },
                )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Path traversal violation: {exc}",
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=f"Path traversal violation: {exc}",
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # The tool ran and raised: it may have written before it did (#1366).
                    writes_files=declared_writes,  # refused partway
                    spawns_subagents=declared_spawns,  # refused partway
                )
            except Exception as exc:
                duration_ms = (asyncio.get_running_loop().time() - t_start) * 1000.0
                logger.warning(
                    "Tool '%s' execution failed with unexpected exception: %s",
                    effective_tc.name,
                    exc,
                    extra={
                        "agent_id": self.agent_id,
                        "session_id": self._context.session_id,
                        "tool_name": effective_tc.name,
                        "error": str(exc),
                    },
                )
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=f"Tool execution failed: {type(exc).__name__}: {exc}",
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )
                rec = ToolExecutionRecord(
                    tool_name=effective_tc.name,
                    arguments=unwrapped_args,
                    output=None,
                    status=ToolResultStatus.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                    tool_call_id=effective_tc.id,
                    # As above: a tool that raised had already started (#1366).
                    writes_files=declared_writes,  # raised partway
                    spawns_subagents=declared_spawns,  # raised partway
                )

        # 2. Execute POST_TOOL_USE hook
        post_ctx = HookContext(
            agent_id=self.agent_id,
            session_id=self._context.session_id,
            trace_id=self._context.trace_id,
            event_type=HookEvent.POST_TOOL_USE,
            payload={
                "tool_name": effective_tc.name,
                "tool_call_id": effective_tc.id,
                "arguments": unwrapped_args,
                "output": rec.output,
                "status": rec.status,
                "error": rec.error,
                "duration_ms": rec.duration_ms,
            },
        )
        post_decision = await self._hook_runner.run_hooks(HookEvent.POST_TOOL_USE, post_ctx)
        if post_decision.action == HookAction.BLOCK:
            block_reason = post_decision.reason or "Blocked by post-tool hook"
            err_msg = f"Tool execution blocked by hook: {block_reason}"
            msg = ChatMessage(
                role=MessageRole.TOOL,
                content=err_msg,
                name=effective_tc.name,
                tool_call_id=effective_tc.id,
            )
            rec = rec.model_copy(update={"status": "error", "error": err_msg, "output": None})
        elif (
            post_decision.action == HookAction.MODIFY and post_decision.modified_payload is not None
        ):
            mod_payload = post_decision.modified_payload
            if "output" in mod_payload:
                new_output = mod_payload["output"]
                rec = rec.model_copy(update={"output": new_output})
                msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=canonical_tool_text(new_output),
                    name=effective_tc.name,
                    tool_call_id=effective_tc.id,
                )

        if stream_callback is not None:
            try:
                out_val = unwrap_immutable(
                    rec.output if rec.status == ToolResultStatus.SUCCESS else rec.error
                )
                res_tool_done = stream_callback(
                    "tool_call",
                    {
                        "tool": effective_tc.name,
                        "args": unwrapped_args,
                        "output": out_val,
                        "status": "completed",
                        "error": rec.error,
                        "duration_ms": rec.duration_ms,
                    },
                )
                if asyncio.iscoroutine(res_tool_done):
                    await res_tool_done
            except Exception:
                logger.debug("Error in stream_callback during tool finish", exc_info=True)

        return msg, rec
