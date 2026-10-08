"""Protocols for LinkML domain ontology management and validation."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from uclone_x.ontology.models import (
    OntologyAxiom,
    OntologyConcept,
    OntologyRelation,
    ValidationResult,
)


@runtime_checkable
class OntologyValidatorProtocol(Protocol):
    """Protocol for high-speed Tier-1 turn validation."""

    def validate_entity(
        self,
        entity_name: str,
        data: dict[str, Any],
    ) -> ValidationResult:
        """Validate an input or output payload against the compiled ontology.

        No latency figure here: issue 2026-09-02-009 moved every numeric target to
        the performance-budgets document, where each is marked unmeasured, so that
        figures stop being frozen into normative text. A docstring is normative text.
        """
        ...


@runtime_checkable
class OntologyEngineProtocol(Protocol):
    """Protocol for managing the evolving domain ontology knowledge graph."""

    def register_entity(self, entity: OntologyConcept) -> None:
        """Register or update an entity schema."""
        ...

    def register_relation(self, relation: OntologyRelation) -> None:
        """Register a relationship schema."""
        ...

    def register_axiom(self, axiom: OntologyAxiom) -> None:
        """Register an axiom invariant rule."""
        ...

    def get_entity(self, name: str) -> OntologyConcept | None:
        """Retrieve entity definition."""
        ...

    def get_concept(self, name: str) -> OntologyConcept | None:
        """Retrieve concept definition."""
        ...

    def get_axiom(self, name: str) -> OntologyAxiom | None:
        """Retrieve axiom definition."""
        ...

    def get_active_invariants(
        self,
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> list[OntologyAxiom]:
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
