"""Data models for agent lifecycle and tool execution hooks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HookEvent(StrEnum):
    """Lifecycle event types that hooks can intercept."""

    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    ON_ERROR = "on_error"

    @classmethod
    def _missing_(cls, value: object) -> HookEvent | None:
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class HookAction(StrEnum):
    """Action returned by a hook evaluation."""

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ASK = "ask"

    @classmethod
    def _missing_(cls, value: object) -> HookAction | None:
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class FailurePolicy(StrEnum):
    """Failure policy determining behavior when a hook execution errors or times out."""

    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"

    @classmethod
    def _missing_(cls, value: object) -> FailurePolicy | None:
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None


class HookDecision(BaseModel):
    """Immutable evaluation outcome from a hook execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    action: HookAction = Field(
        default=HookAction.ALLOW,
        description="The action to take: ALLOW, BLOCK, or MODIFY.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional human-readable explanation for BLOCK or MODIFY.",
    )
    modified_payload: dict[str, Any] | None = Field(
        default=None,
        description="Modified payload dictionary when action is MODIFY.",
    )


class HookContext(BaseModel):
    """Execution context passed to hook handlers."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    agent_id: str = Field(description="Identifier of the executing agent.")
    session_id: str | None = Field(default=None, description="Active session ID if available.")
    trace_id: str | None = Field(default=None, description="Active trace ID if available.")
    event_type: HookEvent = Field(description="The intercepted hook lifecycle event.")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload (e.g. tool arguments, turn input/output, or error data).",
    )


class ApprovalDecision(BaseModel):
    """Decision made by a human approver for a suspended tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    action: HookAction = Field(description="The approval decision: ALLOW, BLOCK, or MODIFY.")
    reason: str | None = Field(default=None, description="Explanation for the decision.")
    modified_arguments: dict[str, Any] | None = Field(
        default=None, description="Modified arguments if action is MODIFY."
    )
    decided_by: str | None = Field(
        default=None, description="Identifier of the user who made the decision."
    )


class ApprovalRequestPayload(BaseModel):
    """Payload published when a tool execution requires human approval."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request_id: str = Field(description="Unique identifier for the approval request.")
    tool_call_id: str = Field(description="ID of the tool call awaiting approval.")
    tool_name: str = Field(description="Name of the tool being called.")
    arguments: dict[str, Any] = Field(description="Arguments for the tool call.")
    agent_id: str = Field(description="ID of the agent making the tool call.")
    session_id: str | None = Field(default=None, description="Session ID if available.")
    prompt: str | None = Field(
        default=None, description="Optional prompt or explanation for the human."
    )
