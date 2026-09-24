"""Data models for self-constructing and evolving LinkML ontologies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uclone_x.core.immutable import ImmutableStrMapping
from uclone_x.core.provenance import Provenance


class OntologyTier(StrEnum):
    """Tier of confidence and enforcement authority for an ontology element."""

    ASSERTED = "asserted"
    ASSERTED_CORE = "asserted_core"
    ASSERTED_DOMAIN = "asserted_domain"
    INDUCED_ENFORCING = "induced_enforcing"
    INDUCED_CANDIDATE = "induced_candidate"

    @classmethod
    def _missing_(cls, value: object) -> OntologyTier | None:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return None


def normalize_tier(val: str | OntologyTier) -> OntologyTier:
    """Normalize and validate an ontology tier string or enum (supporting hyphen and underscore aliases)."""
    if isinstance(val, OntologyTier):
        return val
    try:
        return OntologyTier(val)
    except ValueError as err:
        raise ValueError(
            f"Unrecognised ontology tier: '{val}' (valid: {[t.value for t in OntologyTier]})"
        ) from err


def tier_to_precedence(tier: OntologyTier | str) -> int:
    """Return numeric precedence for an ontology tier (higher wins)."""
    tier_enum = normalize_tier(tier) if not isinstance(tier, OntologyTier) else tier
    if tier_enum in (
        OntologyTier.ASSERTED,
        OntologyTier.ASSERTED_CORE,
        OntologyTier.ASSERTED_DOMAIN,
    ):
        return 100
    if tier_enum == OntologyTier.INDUCED_ENFORCING:
        return 50
    if tier_enum == OntologyTier.INDUCED_CANDIDATE:
        return 10
    raise ValueError(f"Unrecognised ontology tier: '{tier}'")


def utc_now_iso() -> str:
    """Current UTC ISO 8601 formatted timestamp string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_concept_hash(
    name: str,
    parent_type: str | None,
    attributes: Mapping[str, str],
    required_fields: tuple[str, ...] | list[str],
) -> str:
    """Compute canonical deterministic SHA-256 content hash for a concept schema."""
    canonical: dict[str, Any] = {
        "attributes": dict(sorted(attributes.items())),
        "name": name,
        "parent_type": parent_type,
        "required_fields": sorted(required_fields),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_relation_hash(
    source_entity: str,
    predicate: str,
    target_entity: str,
    is_directed: bool,
) -> str:
    """Compute canonical deterministic SHA-256 content hash for a relation schema."""
    canonical: dict[str, Any] = {
        "is_directed": is_directed,
        "predicate": predicate,
        "source_entity": source_entity,
        "target_entity": target_entity,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_axiom_hash(
    name: str,
    subject_entity: str,
    predicate: str,
    object_value: str,
    rule_expression: str,
) -> str:
    """Compute canonical deterministic SHA-256 content hash for an ontology axiom."""
    canonical: dict[str, Any] = {
        "name": name,
        "object_value": object_value,
        "predicate": predicate,
        "rule_expression": rule_expression,
        "subject_entity": subject_entity,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EvidenceRecord(BaseModel):
    """Provenance and observational evidence for induced ontology elements."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    observation_count: int = 1
    first_seen: str = Field(default_factory=utc_now_iso)
    last_seen: str = Field(default_factory=utc_now_iso)
    source_session: str | None = None
    originating_sessions: tuple[str, ...] = Field(default_factory=tuple)
    model_id: str | None = None
    induced_by_provider: str | None = None
    contradicting_observations: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _populate_evidence_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        d: dict[str, Any] = dict(raw)
        if "originating_sessions" in d:
            d["originating_sessions"] = (
                tuple(str(x) for x in cast(Sequence[Any], d["originating_sessions"]))
                if isinstance(d["originating_sessions"], (list, tuple))
                else ()
            )
        elif "source_session" in d and d["source_session"]:
            d["originating_sessions"] = (str(d["source_session"]),)
        if "contradicting_observations" in d:
            d["contradicting_observations"] = (
                tuple(str(x) for x in cast(Sequence[Any], d["contradicting_observations"]))
                if isinstance(d["contradicting_observations"], (list, tuple))
                else ()
            )
        return d

    @property
    def session_count(self) -> int:
        """Count of distinct originating sessions."""
        return len(self.distinct_sessions)

    @property
    def distinct_sessions(self) -> tuple[str, ...]:
        """Tuple of distinct originating session IDs in order of first appearance."""
        seen: set[str] = set()
        result: list[str] = []
        for s in self.originating_sessions:
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        if self.source_session and self.source_session not in seen:
            result.append(self.source_session)
        return tuple(result)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Override model_copy so updates are strictly validated through model_validate (Issue #42)."""
        if update:
            dumped: dict[str, Any] = dict(self.model_dump())
            dumped.update(update)
            return type(self).model_validate(dumped)
        return super().model_copy(deep=deep)


class OntologyConcept(BaseModel):
    """Structured entity definition in the agent's domain ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    iri: str | None = None
    description: str = ""
    parent_type: str | None = None
    attributes: ImmutableStrMapping = Field(default_factory=dict)
    required_fields: tuple[str, ...] = Field(default_factory=tuple)
    tier: OntologyTier = OntologyTier.ASSERTED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: EvidenceRecord | None = None
    content_hash: str = ""
    precedence: int = 100

    @model_validator(mode="before")
    @classmethod
    def _populate_concept_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        d: dict[str, Any] = dict(raw)
        if "tier" in d:
            tier_final = normalize_tier(d["tier"])
        else:
            tier_final = OntologyTier.ASSERTED
        d["tier"] = tier_final

        if "precedence" not in d or d.get("precedence") == 0:
            d["precedence"] = tier_to_precedence(tier_final)

        name = str(d.get("name", ""))
        parent_raw = d.get("parent_type")
        parent: str | None = str(parent_raw) if parent_raw is not None else None
        attrs_raw = d.get("attributes", {})
        attrs: dict[str, str] = (
            {str(k): str(v) for k, v in cast(Mapping[Any, Any], attrs_raw).items()}
            if isinstance(attrs_raw, Mapping)
            else {}
        )
        if "attributes" in d and isinstance(d["attributes"], Mapping):
            d["attributes"] = attrs

        req_raw = d.get("required_fields", ())
        req: tuple[str, ...] = (
            tuple(str(x) for x in cast(Sequence[Any], req_raw))
            if isinstance(req_raw, (list, tuple))
            else ()
        )
        if "required_fields" in d and isinstance(d["required_fields"], (list, tuple)):
            d["required_fields"] = req

        ev_raw = d.get("evidence")
        if isinstance(ev_raw, dict):
            d["evidence"] = EvidenceRecord.model_validate(ev_raw)

        d["content_hash"] = compute_concept_hash(name, parent, attrs, req)
        return d

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Override model_copy so updates are strictly validated through model_validate and content_hash recomputed (Issue #42)."""
        if update:
            dumped: dict[str, Any] = dict(self.model_dump())
            dumped.update(update)
            return type(self).model_validate(dumped)
        return super().model_copy(deep=deep)


class OntologyRelation(BaseModel):
    """Relationship triplet between domain entities."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_entity: str
    predicate: str
    target_entity: str
    is_directed: bool = True
    tier: OntologyTier = OntologyTier.ASSERTED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: EvidenceRecord | None = None
    content_hash: str = ""
    precedence: int = 100

    @model_validator(mode="before")
    @classmethod
    def _populate_relation_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        d: dict[str, Any] = dict(raw)
        if "tier" in d:
            tier_final = normalize_tier(d["tier"])
        else:
            tier_final = OntologyTier.ASSERTED
        d["tier"] = tier_final

        if "precedence" not in d or d.get("precedence") == 0:
            d["precedence"] = tier_to_precedence(tier_final)

        ev_raw = d.get("evidence")
        if isinstance(ev_raw, dict):
            d["evidence"] = EvidenceRecord.model_validate(ev_raw)

        src = str(d.get("source_entity", ""))
        pred = str(d.get("predicate", ""))
        tgt = str(d.get("target_entity", ""))
        directed = bool(d.get("is_directed", True))
        d["content_hash"] = compute_relation_hash(src, pred, tgt, directed)
        return d

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Override model_copy so updates are strictly validated through model_validate and content_hash recomputed (Issue #42)."""
        if update:
            dumped: dict[str, Any] = dict(self.model_dump())
            dumped.update(update)
            return type(self).model_validate(dumped)
        return super().model_copy(deep=deep)


class OntologyAxiom(BaseModel):
    """Axiom or invariant rule asserted or induced for an entity concept."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    subject_entity: str
    predicate: str = ""
    object_value: str = ""
    rule_expression: str = ""
    description: str = ""
    domain: str | None = None
    tier: OntologyTier = OntologyTier.ASSERTED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: EvidenceRecord | None = None
    content_hash: str = ""
    precedence: int = 100

    @model_validator(mode="before")
    @classmethod
    def _populate_axiom_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        d: dict[str, Any] = dict(raw)
        if "tier" in d:
            tier_final = normalize_tier(d["tier"])
        else:
            tier_final = OntologyTier.ASSERTED
        d["tier"] = tier_final

        if "precedence" not in d or d.get("precedence") == 0:
            d["precedence"] = tier_to_precedence(tier_final)

        ev_raw = d.get("evidence")
        if isinstance(ev_raw, dict):
            d["evidence"] = EvidenceRecord.model_validate(ev_raw)

        name = str(d.get("name", ""))
        subj = str(d.get("subject_entity", ""))
        pred = str(d.get("predicate", ""))
        obj = str(d.get("object_value", ""))
        expr = str(d.get("rule_expression", ""))
        d["content_hash"] = compute_axiom_hash(name, subj, pred, obj, expr)
        return d

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Override model_copy so updates are strictly validated through model_validate and content_hash recomputed (Issue #42)."""
        if update:
            dumped: dict[str, Any] = dict(self.model_dump())
            dumped.update(update)
            return type(self).model_validate(dumped)
        return super().model_copy(deep=deep)


class ValidationResult(BaseModel):
    """Result of Tier-1 inline Pydantic/LinkML constraint validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    is_valid: bool
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    matched_tier: OntologyTier | None = None
    content_hash: str | None = None
    latency_ms: float = 0.0
    provenance: Provenance | None = None


# Backward-compatibility aliases
EntitySchema = OntologyConcept
RelationSchema = OntologyRelation
OntologyValidationResult = ValidationResult
OntologyInvariant = OntologyAxiom
InvariantSchema = OntologyAxiom
