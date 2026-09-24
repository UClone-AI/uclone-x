"""Permission modes and human approval hooks for agent tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from uclone_x.agent.hooks.models import HookAction, HookContext, HookDecision
from uclone_x.agent.hooks.protocols import BaseHook

#: Name prefixes of tools that can change the host. A floor under the metadata check in
#: `HumanApprovalHook._is_destructive`, for a call whose tool the agent could not resolve.
_DESTRUCTIVE_PREFIXES = ("execute_", "shell", "write_", "edit_", "delete_", "run_")

#: Registered tools that change the host and that no prefix above catches: the shell the
#: model is offered (`bash_run`; `run_command` is its unadvertised alias since #1461) and
#: the two file-writing tools (#1463).
_DESTRUCTIVE_TOOL_NAMES = frozenset({"bash_run", "file_write", "file_edit"})


class PermissionMode(StrEnum):
    """Governs what tools require human approval. Orthogonal to sandbox isolation."""

    DEFAULT = "default"
    AUTO = "auto"
    PLAN = "plan"


class HumanApprovalHook(BaseHook):
    """Hook that suspends tool execution to await human approval based on permission mode."""

    def __init__(
        self,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        ask_tools: set[str] | None = None,
        blocked_tools: set[str] | None = None,
    ) -> None:
        self.permission_mode = permission_mode
        self._ask_tools = ask_tools or set()
        self._blocked_tools = blocked_tools or set()

    @property
    def name(self) -> str:
        return "human_approval_hook"

    def _is_destructive(self, tool_name: str, payload: Mapping[str, Any]) -> bool:
        """Whether a call to `tool_name` needs approval, from the tool's metadata first.

        The agent puts the resolved tool's own declarations in the `PRE_TOOL_USE` payload
        (`writes_files`, `spawns_subagents`; see `uclone_x.tools.base`). A tool that writes
        a file or starts another agent is destructive whatever it is called: a name is not a
        capability, and the name prefixes alone missed `bash_run`, the one shell the model is
        offered since #1461, and `file_write` (#1463).

        The name check stays as a floor, not a fallback only. It covers a call to a tool the
        agent could not resolve (no metadata in the payload), and a payload claiming a shell
        writes nothing still asks.

        The payload is trusted because of where it comes from, not because of this method:
        under `HookRunner`, `tool_name` and the declarations are exactly what the agent
        resolved from the call it will run, since an earlier hook cannot rewrite them
        (`_PRE_TOOL_USE_FIXED_KEYS`, #1488). Dispatched outside a runner, the caller vouches
        for them.
        """
        if payload.get("writes_files") is True or payload.get("spawns_subagents") is True:
            return True
        return tool_name in _DESTRUCTIVE_TOOL_NAMES or tool_name.startswith(_DESTRUCTIVE_PREFIXES)

    async def dispatch(self, context: HookContext) -> HookDecision:
        tool_name = context.payload.get("tool_name", "")

        if self._blocked_tools and tool_name in self._blocked_tools:
            return HookDecision(
                action=HookAction.BLOCK, reason=f"Tool '{tool_name}' is explicitly blocked."
            )

        if self.permission_mode == PermissionMode.AUTO:
            return HookDecision(action=HookAction.ALLOW)

        is_destructive = self._is_destructive(tool_name, context.payload)
        in_ask_tools = tool_name in self._ask_tools

        if self.permission_mode == PermissionMode.PLAN:
            if is_destructive:
                return HookDecision(
                    action=HookAction.BLOCK, reason="Write tools blocked in plan mode"
                )
            return HookDecision(action=HookAction.ALLOW)

        if self.permission_mode == PermissionMode.DEFAULT:
            if is_destructive or in_ask_tools:
                return HookDecision(action=HookAction.ASK, reason="Human approval required")
            return HookDecision(action=HookAction.ALLOW)

        return HookDecision(action=HookAction.ALLOW)
