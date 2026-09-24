"""Data models for structured, typed cross-session memory facts adhering to Principle 6."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.provenance import Provenance
from uclone_x.errors import MissingProvenanceError


def utc_now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryFact(BaseModel):
    """Structured, typed memory fact carrying in-band P6 provenance.

    Avoids raw, unverified free-text memory dumps by strictly typing
    facts into (subject, predicate, object_value) tuples with explicit
    confidence, provenance, and retraction metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fact_id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for the memory fact.",
    )
    subject: str = Field(
        min_length=1,
        description="The entity, topic, or domain concept this fact relates to.",
    )
    predicate: str = Field(
        min_length=1,
        description="The relation, attribute, or property asserted.",
    )
    object_value: str = Field(
        min_length=1,
        description="The asserted value or content.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in range [0.0, 1.0].",
    )
    provenance: Provenance = Field(
        description="Principle 6 in-band provenance establishing attribution.",
    )
    source_session_id: str = Field(
        min_length=1,
        description="Session ID where this fact was originally established or observed.",
    )
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO 8601 timestamp when fact was recorded.",
    )
    updated_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO 8601 timestamp of last status update.",
    )
    retracted: bool = Field(
        default=False,
        description="Whether this fact has been explicitly retracted.",
    )
    retraction_reason: str | None = Field(
        default=None,
        description="Explanation for fact retraction if retracted.",
    )
    retracted_at: str | None = Field(
        default=None,
        description="UTC timestamp when fact was retracted.",
    )
    contradicts_fact_id: str | None = Field(
        default=None,
        description="ID of a prior fact this fact contradicts or supersedes.",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorization and scoping tags for selective retrieval.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata attributes.",
    )

    @field_validator("provenance", mode="before")
    @classmethod
    def _enforce_provenance(cls, v: Any) -> Any:
        if v is None:
            raise MissingProvenanceError(
                "MemoryFact carries no provenance; refusing to treat it as a primary result"
            )
        return v

    def conflicts_with(self, other: MemoryFact) -> bool:
        """Determine if this fact conflicts with another active fact.

        A conflict occurs when two active facts share the same subject and predicate
        (normalized) but assert different object values.
        """
        if self.retracted or other.retracted:
            return False
        if self.fact_id == other.fact_id:
            return False

        same_subject = self.subject.strip().lower() == other.subject.strip().lower()
        same_predicate = self.predicate.strip().lower() == other.predicate.strip().lower()
        different_value = self.object_value.strip() != other.object_value.strip()

        return same_subject and same_predicate and different_value

    def summary(self) -> str:
        """Format fact as concise readable string for progressive prompt disclosure."""
        return f"{self.subject}: {self.predicate} -> {self.object_value}"


class RetractionRecord(BaseModel):
    """Immutable audit record documenting a fact retraction."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fact_id: str = Field(description="ID of the retracted fact.")
    reason: str = Field(description="Reason for retraction.")
    retracted_at: str = Field(default_factory=utc_now_iso)
    retracted_by_session: str | None = Field(default=None)
    provenance: Provenance = Field(description="Provenance of the retraction action.")

    @field_validator("provenance", mode="before")
    @classmethod
    def _enforce_provenance(cls, v: Any) -> Any:
        if v is None:
            raise MissingProvenanceError(
                "RetractionRecord carries no provenance; refusing to treat it as a primary result"
            )
        return v
