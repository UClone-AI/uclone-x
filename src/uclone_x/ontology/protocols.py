"""Protocols for LinkML domain ontology management and induction."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from uclone_x.ontology.models import (
    EntitySchema,
    OntologyAxiom,
    OntologyInvariant,
    OntologyValidationResult,
    RelationSchema,
)
from uclone_x.tools.models import ToolResult


@runtime_checkable
class OntologyValidatorProtocol(Protocol):
    """Protocol for high-speed Tier-1 turn validation."""

    def validate_entity(
        self,
        entity_name: str,
        data: dict[str, Any],
    ) -> OntologyValidationResult:
        """Validate an input or output payload against the compiled ontology.

        No latency figure here: issue 2026-09-02-009 moved every numeric target to
        `docs/nfr-performance-budgets.md`, where each is marked unmeasured, so that
        figures stop being frozen into normative text. A docstring is normative text.
        """
        ...


@runtime_checkable
class OntologyInducerProtocol(Protocol):
    """Protocol for autonomous entity and relation extraction from turns."""

    async def induce_from_turn(
        self,
        turn_text: str,
        tool_results: tuple[ToolResult, ...],
    ) -> tuple[EntitySchema, ...]:
        """Extract candidate entities and constraints from successful turns."""
        ...


@runtime_checkable
class OntologyEngineProtocol(Protocol):
    """Protocol for managing the evolving domain ontology knowledge graph."""

    def register_entity(self, entity: EntitySchema) -> None:
        """Register or update an entity schema."""
        ...

    def register_relation(self, relation: RelationSchema) -> None:
        """Register a relationship schema."""
        ...

    def register_axiom(self, axiom: OntologyAxiom) -> None:
        """Register an axiom invariant rule."""
        ...

    def get_entity(self, name: str) -> EntitySchema | None:
        """Retrieve entity definition."""
        ...

    def get_concept(self, name: str) -> EntitySchema | None:
        """Retrieve concept definition."""
        ...

    def get_axiom(self, name: str) -> OntologyAxiom | None:
        """Retrieve axiom definition."""
        ...

    def get_active_invariants(
        self,
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> list[OntologyInvariant]:
        """Retrieve active invariant rules filtered by domain and ontology tier (P7, P8)."""
        ...

    def export_linkml_yaml(self) -> str:
        """Serialize current ontology to LinkML YAML format."""
        ...

    def export_graph(self) -> dict[str, Any]:
        """Export all active concepts, axioms, relations, and hierarchy nodes in JSON structure."""
        ...


# Protocol alias for OntologyEngineProtocol
OntologyServiceProtocol = OntologyEngineProtocol
