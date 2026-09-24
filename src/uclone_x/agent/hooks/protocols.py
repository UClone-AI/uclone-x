"""Protocols and base classes for agent lifecycle and tool execution hooks."""

from __future__ import annotations

import logging

from uclone_x.agent.hooks.models import (
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
)

logger = logging.getLogger(__name__)


class BaseHook:
    """Base class for in-process or external agent hooks."""

    def __init__(
        self,
        name: str | None = None,
        failure_policy: FailurePolicy = FailurePolicy.FAIL_OPEN,
    ) -> None:
        self._name = name or self.__class__.__name__
        self._failure_policy = failure_policy

    @property
    def name(self) -> str:
        """Name or identifier of the hook."""
        return self._name

    @property
    def failure_policy(self) -> FailurePolicy:
        """Failure policy when hook execution fails or times out."""
        return self._failure_policy

    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        """Hook triggered before an agent reasoning turn begins."""
        return HookDecision(action=HookAction.ALLOW)

    async def on_post_turn(self, context: HookContext) -> HookDecision:
        """Hook triggered after an agent reasoning turn finishes successfully."""
        return HookDecision(action=HookAction.ALLOW)

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        """Hook triggered immediately before a tool is invoked."""
        return HookDecision(action=HookAction.ALLOW)

    async def on_post_tool_use(self, context: HookContext) -> HookDecision:
        """Hook triggered immediately after a tool returns its result."""
        return HookDecision(action=HookAction.ALLOW)

    async def on_error(self, context: HookContext) -> HookDecision:
        """Hook triggered when an agent turn or execution encounters an error."""
        return HookDecision(action=HookAction.ALLOW)

    async def dispatch(self, context: HookContext) -> HookDecision:
        """Dispatch the hook context to the corresponding event handler method."""
        if context.event_type == HookEvent.PRE_TURN:
            return await self.on_pre_turn(context)
        elif context.event_type == HookEvent.POST_TURN:
            return await self.on_post_turn(context)
        elif context.event_type == HookEvent.PRE_TOOL_USE:
            return await self.on_pre_tool_use(context)
        elif context.event_type == HookEvent.POST_TOOL_USE:
            return await self.on_post_tool_use(context)
        elif context.event_type == HookEvent.ON_ERROR:
            return await self.on_error(context)
        return HookDecision(action=HookAction.ALLOW)
