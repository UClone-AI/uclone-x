"""Data models for structured, typed cross-session memory facts adhering to Principle 6."""

from __future__ import annotations

import unicodedata
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.provenance import Provenance
from uclone_x.errors import MissingProvenanceError
from uclone_x.knowledge.fold import fold
from uclone_x.knowledge.policy import PERSON_ENTITY, PROJECT_ENTITY, SELF_ENTITY

#: How a fact came to be known (clone-knowledge-graph design §3.2). `told`: extracted from
#: what a person wrote. `found`: extracted from a tool result the clone received, or from
#: what the clone itself said in the turn (then `metadata["grounded_in"]` is `"self"`, #1404).
#: `saved`: the model called `record_memory_fact`. `corrected`: a person edited it.
FactOrigin = Literal["told", "found", "saved", "corrected"]


#: The one subject every fact about the person (the human owner) is filed under
#: (clone-knowledge-graph design §3.3 step 5); recall always includes these (§3.5).
PERSON_SUBJECT = PERSON_ENTITY
#: The subject for durable working preferences of this workspace, such as the reply
#: language or a code style. Recall always includes a few of these, after the person's
#: (clone-knowledge-graph design §3.5); the extractor and `record_memory_fact` name it.
PROJECT_SUBJECT = PROJECT_ENTITY
#: Words a model may use for the person. With the person's own names, all are filed under
#: `PERSON_SUBJECT`, by the extractor and by `record_memory_fact` alike.
PERSON_WORDS = frozenset({"user", "the user", "person", "the person", "human", "me", "i", "owner"})
#: The subject every fact about the clone itself is filed under: its looks, age,
#: personality, speech style, name (#2016). Recall always includes these, first, and says
#: they override the persona description where the two disagree.
SELF_SUBJECT = SELF_ENTITY
#: Words a model may use for the clone itself. With the clone's own names, all are filed
#: under `SELF_SUBJECT`. Second person means the clone because both write paths address
#: the clone as "you": the extractor labels the clone's lines "(you)" and names the person
#: "user", and `record_memory_fact` is called by the clone. "me" and "i" stay the person's
#: (`PERSON_WORDS`): in the extractor they are a person line's own first person.
SELF_WORDS = frozenset(
    {
        "self",
        "itself",
        "you",
        "yourself",
        "clone",
        "the clone",
        "assistant",
        "the assistant",
        "너",
        "넌",
        "너는",
        "네가",
        "니가",
        "당신",
        "당신은",
    }
)


def fold_name(name: str) -> str:
    """`name` as names are compared: in NFC, whitespace collapsed, casefolded (#1893).

    NFC first, so a name stored decomposed ("José" as `e` and a combining accent) matches
    the composed form a model writes, and the reverse.
    """
    return fold(name)


def person_subject(subject: str, person_names: Iterable[str] = ()) -> str:
    """`subject` in NFC with its whitespace collapsed, or `PERSON_SUBJECT` when it names the person.

    It names the person when, folded (`fold_name`), it is one of `PERSON_WORDS` or one of
    `person_names` (the person's id, display name and aliases).
    """
    collapsed = " ".join(unicodedata.normalize("NFC", subject).split())
    names = {fold_name(name) for name in person_names if name.strip()}
    if fold_name(collapsed) in PERSON_WORDS | names:
        return PERSON_SUBJECT
    return collapsed


def fact_subject(
    subject: str, person_names: Iterable[str] = (), clone_names: Iterable[str] = ()
) -> str:
    """`subject` as a fact is filed: `user` for the person, `self` for the clone, else as given.

    The person is checked first (`person_subject`), so a name that means both, which the
    room already leaves out of the person's names when another participant goes by it,
    never turns a fact about the person into one about the clone. Otherwise the subject is
    `SELF_SUBJECT` when, folded (`fold_name`), it is one of `SELF_WORDS` or one of
    `clone_names` (the clone's id, display name and aliases).
    """
    filed = person_subject(subject, person_names)
    names = {fold_name(name) for name in clone_names if name.strip()}
    if fold_name(filed) in SELF_WORDS | names:
        return SELF_SUBJECT
    return filed


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
    origin: FactOrigin = Field(
        default="saved",
        description="How the fact came to be known. Defaults to `saved` so a row written "
        "before this field existed, which has no `origin` key, loads as the only kind of "
        "fact there was then: one the model saved with `record_memory_fact`.",
    )
    source_room_id: str | None = Field(
        default=None,
        description="The conversation (room) the fact was learned in. `None` outside a "
        "conversation, and for rows written before this field existed.",
    )
    source_turn_id: str | None = Field(
        default=None,
        description="The room turn the fact was learned in, the id the room's row carries. "
        "`None` outside a room turn, and for rows written before this field existed.",
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
    valid_until: str | None = Field(
        default=None,
        description="UTC ISO 8601 time the fact stopped holding, when the person said so "
        '("until last year"). `None` for a fact with no stated end. A fact with an end is '
        "history: it is listed and shown with its end, is not worked out from, and neither "
        "supersedes nor is superseded by another fact on the same subject and predicate.",
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

        A conflict occurs when two active facts share the same subject and predicate but
        assert different object values. Subject and predicate are compared folded
        (`fold_name`, NFC included), so a subject stored decomposed before #1895 still
        meets the composed form a model writes now (#1893).
        """
        if self.retracted or other.retracted:
            return False
        if self.valid_until is not None or other.valid_until is not None:
            return False  # a stated end makes it history, which a current value does not replace
        if self.fact_id == other.fact_id:
            return False

        same_subject = fold_name(self.subject) == fold_name(other.subject)
        same_predicate = fold_name(self.predicate) == fold_name(other.predicate)
        different_value = self.object_value.strip() != other.object_value.strip()

        return same_subject and same_predicate and different_value

    def summary(self) -> str:
        """Format fact as concise readable string for progressive prompt disclosure."""
        said = f"{self.subject}: {self.predicate} -> {self.object_value}"
        return said if self.valid_until is None else f"{said} (until {self.valid_until[:10]})"


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
