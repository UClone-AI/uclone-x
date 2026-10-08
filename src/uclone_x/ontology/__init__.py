"""Ontology subsystem: LinkML domain schemas, Tier-1 validation, tiers and reasoning."""

# The engine and reasoner are re-exported lazily (PEP 562), as `uclone_x.llm` does its
# connectors (#1775). The agent kernel reads `ontology.protocols`, which runs this file
# first, so eager re-exports loaded the engine, the reasoner and the rule evaluator into
# every agent that has no ontology (#2012). `from uclone_x.ontology import OntologyEngine`
# still works, on first use.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from uclone_x.ontology.models import (
    EvidenceRecord,
    OntologyAxiom,
    OntologyConcept,
    OntologyRelation,
    OntologyTier,
    ValidationResult,
    compute_axiom_hash,
    compute_concept_hash,
    compute_relation_hash,
)
from uclone_x.ontology.protocols import (
    OntologyEngineProtocol,
    OntologyServiceProtocol,
    OntologyValidatorProtocol,
)

if TYPE_CHECKING:
    from uclone_x.ontology.engine import (
        ExpressionEvaluationError,
        OntologyEngine,
        OntologyService,
        safe_eval_rule_expression,
    )
    from uclone_x.ontology.justification import (
        TYPE_PREDICATE,
        Closure,
        Fact,
        Inconsistency,
        ProofStep,
        UnsupportedAxiom,
        compute_fact_id,
    )
    from uclone_x.ontology.reasoner import (
        OntologyReasoner,
        check_consistency,
        materialize,
    )

#: Each lazily re-exported name -> the module that defines it, loaded on first access.
_LAZY_NAMES: dict[str, str] = {
    "ExpressionEvaluationError": "uclone_x.ontology.engine",
    "OntologyEngine": "uclone_x.ontology.engine",
    "OntologyService": "uclone_x.ontology.engine",
    "safe_eval_rule_expression": "uclone_x.ontology.engine",
    "TYPE_PREDICATE": "uclone_x.ontology.justification",
    "Closure": "uclone_x.ontology.justification",
    "Fact": "uclone_x.ontology.justification",
    "Inconsistency": "uclone_x.ontology.justification",
    "ProofStep": "uclone_x.ontology.justification",
    "UnsupportedAxiom": "uclone_x.ontology.justification",
    "compute_fact_id": "uclone_x.ontology.justification",
    "OntologyReasoner": "uclone_x.ontology.reasoner",
    "check_consistency": "uclone_x.ontology.reasoner",
    "materialize": "uclone_x.ontology.reasoner",
}


def __getattr__(name: str) -> Any:
    """Load an adapter name on first access, so importing this package loads no adapter."""
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "Closure",
    "EvidenceRecord",
    "ExpressionEvaluationError",
    "Fact",
    "Inconsistency",
    "OntologyAxiom",
    "OntologyConcept",
    "OntologyEngine",
    "OntologyEngineProtocol",
    "OntologyReasoner",
    "OntologyRelation",
    "OntologyService",
    "OntologyServiceProtocol",
    "OntologyTier",
    "OntologyValidatorProtocol",
    "ProofStep",
    "TYPE_PREDICATE",
    "UnsupportedAxiom",
    "ValidationResult",
    "check_consistency",
    "compute_axiom_hash",
    "compute_concept_hash",
    "compute_fact_id",
    "compute_relation_hash",
    "materialize",
    "safe_eval_rule_expression",
]
