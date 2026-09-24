"""Data models for Google Agent-to-Agent (A2A) protocol v1.0.1."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.immutable import ImmutableJsonMapping, ImmutableStrMapping
from uclone_x.core.provenance import Provenance


class WireProtocolType(StrEnum):
    """Transport protocol type."""

    LOCAL_IN_MEMORY = "local_in_memory"
    REST_SSE = "rest_sse"


class TaskStatus(StrEnum):
    """Lifecycle and execution status of an A2A task per A2A v1.0.1 specification."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


class AgentCard(BaseModel):
    """Agent discovery metadata exposed at /.well-known/agent-card.json per RFC 8615."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    description: str
    version: str = "1.0.1"
    endpoints: ImmutableStrMapping = Field(default_factory=dict)
    skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Declared skills and capabilities of the agent per A2A v1.0.1.",
    )
    input_schema: ImmutableJsonMapping = Field(default_factory=dict)
    output_schema: ImmutableJsonMapping = Field(default_factory=dict)


class TaskMessage(BaseModel):
    """A2A task dispatch envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    task_id: str
    session_id: str
    input_data: ImmutableJsonMapping = Field(default_factory=dict)
    sender_agent_id: str
    target_agent_id: str
    metadata: ImmutableStrMapping = Field(default_factory=dict)


class TaskResult(BaseModel):
    """A2A task response envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    output_data: ImmutableJsonMapping = Field(default_factory=dict)
    error: str | None = None
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )
