"""Hook runner managing lifecycle dispatch, modification aggregation, and short-circuit evaluation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from uclone_x.agent.hooks.models import (
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
)
from uclone_x.agent.hooks.protocols import BaseHook
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventPriority,
    EventType,
)
from uclone_x.engine.protocols import (
    EventBusProtocol,
    PublisherHandleProtocol,
)

logger = logging.getLogger(__name__)

#: `PRE_TOOL_USE` payload keys that describe the call the agent will run: which tool, which
#: call, and what that tool declares about itself. The agent runs the original call's tool
#: whatever a hook returns -- only `arguments` from a `MODIFY` reaches execution -- so a
#: rewrite of these keys would change what later hooks judge and not what runs (#1488).
#: `run_hooks` therefore restores them before every hook sees the payload.
_PRE_TOOL_USE_FIXED_KEYS = ("tool_name", "tool_call_id", "writes_files", "spawns_subagents")


def _with_fixed_keys(payload: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    """A copy of `payload` whose fixed keys are exactly `fixed`: rewrites and additions go."""
    kept = {k: v for k, v in payload.items() if k not in _PRE_TOOL_USE_FIXED_KEYS}
    return kept | fixed


class HookRunner:
    """Orchestrates registration and sequential execution of lifecycle and tool hooks."""

    def __init__(
        self,
        hooks: Sequence[BaseHook] | None = None,
        bus: EventBusProtocol | None = None,
        publisher: PublisherHandleProtocol | None = None,
    ) -> None:
        self._hooks: list[BaseHook] = list(hooks) if hooks is not None else []
        self._bus = bus
        self._publisher = publisher

    @property
    def hooks(self) -> tuple[BaseHook, ...]:
        """Snapshot of currently registered hooks."""
        return tuple(self._hooks)

    def register_hook(self, hook: BaseHook) -> None:
        """Register a new hook into the runner."""
        self._hooks.append(hook)

    def register_hooks(self, hooks: Sequence[BaseHook]) -> None:
        """Register multiple hooks into the runner."""
        self._hooks.extend(hooks)

    async def _publish_event(
        self,
        event_type: EventType,
        hook_name: str,
        lifecycle_event: HookEvent,
        decision: HookDecision,
        context: HookContext,
    ) -> None:
        """Publish a hook execution event to the EventBus if wired."""
        if self._bus is None and self._publisher is None:
            return
        payload: dict[str, Any] = {
            "hook_name": hook_name,
            "lifecycle_event": lifecycle_event.value,
            "action": decision.action.value,
            "reason": decision.reason,
            "agent_id": context.agent_id,
        }
        if context.session_id:
            payload["session_id"] = context.session_id
        evt = AgentEvent(
            type=event_type,
            topic=f"session.{context.session_id}" if context.session_id else "agent.hooks",
            sender_id=context.agent_id,
            priority=EventPriority.NORMAL,
            payload=payload,
            trace_id=context.trace_id,
        )
        try:
            if self._publisher is not None:
                await self._publisher.publish(evt)
            elif self._bus is not None:
                await self._bus.publish(evt)
        except Exception:
            logger.debug(
                "Failed to publish hook event %s for %s", event_type.value, hook_name, exc_info=True
            )

    async def run_hooks(
        self,
        event_type: HookEvent,
        context: HookContext,
    ) -> HookDecision:
        """Execute registered hooks for a given event, short-circuiting on BLOCK.

        Aggregates MODIFY payloads sequentially and passes updated context to subsequent hooks.
        For `PRE_TOOL_USE`, the keys naming the call (`_PRE_TOOL_USE_FIXED_KEYS`) are not
        modifiable: every hook sees them as the agent sent them, so a hook placed later --
        `HumanApprovalHook` above all -- judges the call that will actually run (#1488).
        """
        if not self._hooks:
            return HookDecision(action=HookAction.ALLOW)

        active_payload = dict(context.payload)
        current_context = (
            context
            if context.event_type == event_type
            else context.model_copy(update={"event_type": event_type, "payload": active_payload})
        )
        modified = False
        # What the call is, as the agent built it, so that no hook -- by a `MODIFY` or by
        # mutating the dict it was handed -- can make a later hook judge another call.
        fixed: dict[str, Any] | None = None
        if event_type == HookEvent.PRE_TOOL_USE:
            fixed = {
                k: context.payload[k] for k in _PRE_TOOL_USE_FIXED_KEYS if k in context.payload
            }

        for hook in self._hooks:
            if fixed is not None:
                current_context = current_context.model_copy(
                    update={"payload": _with_fixed_keys(active_payload, fixed)}
                )
            try:
                decision = await hook.dispatch(current_context)
            except Exception as exc:
                if hook.failure_policy == FailurePolicy.FAIL_CLOSED:
                    err_msg = f"Hook '{hook.name}' raised unhandled exception: {exc} (fail_closed)"
                    logger.warning("%s", err_msg)
                    decision = HookDecision(action=HookAction.BLOCK, reason=err_msg)
                    await self._publish_event(
                        EventType.HOOK_FAILED, hook.name, event_type, decision, current_context
                    )
                else:
                    err_msg = f"Hook '{hook.name}' raised unhandled exception: {exc} (fail_open)"
                    logger.warning("%s", err_msg)
                    decision = HookDecision(action=HookAction.ALLOW, reason=err_msg)
                    await self._publish_event(
                        EventType.HOOK_FAILED, hook.name, event_type, decision, current_context
                    )

            if decision.action in (HookAction.BLOCK, HookAction.ASK):
                await self._publish_event(
                    EventType.HOOK_EXECUTED, hook.name, event_type, decision, current_context
                )
                return decision

            if decision.action == HookAction.MODIFY and decision.modified_payload:
                active_payload.update(decision.modified_payload)
                current_context = current_context.model_copy(update={"payload": active_payload})
                modified = True

            await self._publish_event(
                EventType.HOOK_EXECUTED, hook.name, event_type, decision, current_context
            )

        if modified:
            if fixed is not None:
                active_payload = _with_fixed_keys(active_payload, fixed)
            return HookDecision(action=HookAction.MODIFY, modified_payload=active_payload)

        return HookDecision(action=HookAction.ALLOW)
