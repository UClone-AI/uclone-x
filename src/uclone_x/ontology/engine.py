"""Tiered Ontology Engine with content-hash deterministic validation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    OntologyContradictionError,
    OntologyPromotionError,
    OntologyRetractionBlockedError,
    OntologyViolationError,
    PromotionCriteriaNotMetError,
    UnparseableDirectiveError,
)
from uclone_x.llm.models import ChatMessage
from uclone_x.ontology.models import (
    EvidenceRecord,
    OntologyAxiom,
    OntologyConcept,
    OntologyRelation,
    OntologyTier,
    ValidationResult,
    normalize_tier,
    tier_to_precedence,
    utc_now_iso,
)
from uclone_x.tools.models import ToolResult


def _slugify_axiom_name(raw: str, max_len: int = 60) -> str:
    """Generate a clean slug identifier from rule text."""
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    if not clean:
        return f"rule_{int(time.time())}"
    return clean[:max_len].rstrip("_")


class ExpressionEvaluationError(ValueError):
    """Raised when an ontology axiom rule expression cannot be safely evaluated or fails validation."""


class _SafeAstEvaluator:
    """Safe AST-based expression evaluator supporting comparison, boolean, and arithmetic operations."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        self._context = context

    def eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.eval_node(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in self._context:
                return self._context[node.id]
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id == "None":
                return None
            raise ExpressionEvaluationError(f"Identifier '{node.id}' not found in payload context.")

        if isinstance(node, ast.UnaryOp):
            operand_val: Any = self.eval_node(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand_val
            if isinstance(node.op, ast.UAdd):
                return +operand_val
            if isinstance(node.op, ast.USub):
                return -operand_val
            raise ExpressionEvaluationError(f"Unsupported unary operator: {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            left_val: Any = self.eval_node(node.left)
            right_val: Any = self.eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left_val + right_val
            if isinstance(node.op, ast.Sub):
                return left_val - right_val
            if isinstance(node.op, ast.Mult):
                return left_val * right_val
            if isinstance(node.op, ast.Div):
                if right_val == 0:
                    raise ExpressionEvaluationError("Division by zero in rule expression.")
                return left_val / right_val
            if isinstance(node.op, ast.FloorDiv):
                if right_val == 0:
                    raise ExpressionEvaluationError("Floor division by zero in rule expression.")
                return left_val // right_val
            if isinstance(node.op, ast.Mod):
                if right_val == 0:
                    raise ExpressionEvaluationError("Modulo by zero in rule expression.")
                return left_val % right_val
            if isinstance(node.op, ast.Pow):
                if isinstance(right_val, (int, float)) and (right_val > 1000 or right_val < -1000):
                    raise ExpressionEvaluationError(
                        "Exponent out of supported range in rule expression."
                    )
                return left_val**right_val
            raise ExpressionEvaluationError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for val_node in node.values:
                    if not self.eval_node(val_node):
                        return False
                return True
            if isinstance(node.op, ast.Or):
                for val_node in node.values:
                    if self.eval_node(val_node):
                        return True
                return False
            raise ExpressionEvaluationError(
                f"Unsupported boolean operator: {type(node.op).__name__}"
            )

        if isinstance(node, ast.Compare):
            left_val = self.eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right_val = self.eval_node(comparator)
                matched: bool = False
                if isinstance(op, ast.Eq):
                    matched = left_val == right_val
                elif isinstance(op, ast.NotEq):
                    matched = left_val != right_val
                elif isinstance(op, ast.Lt):
                    matched = left_val < right_val
                elif isinstance(op, ast.LtE):
                    matched = left_val <= right_val
                elif isinstance(op, ast.Gt):
                    matched = left_val > right_val
                elif isinstance(op, ast.GtE):
                    matched = left_val >= right_val
                elif isinstance(op, ast.In):
                    matched = left_val in right_val
                elif isinstance(op, ast.NotIn):
                    matched = left_val not in right_val
                elif isinstance(op, ast.Is):
                    matched = left_val is right_val
                elif isinstance(op, ast.IsNot):
                    matched = left_val is not right_val
                else:
                    raise ExpressionEvaluationError(
                        f"Unsupported comparison operator: {type(op).__name__}"
                    )
                if not matched:
                    return False
                left_val = right_val
            return True

        if isinstance(node, ast.List):
            return [self.eval_node(elt) for elt in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self.eval_node(elt) for elt in node.elts)

        if isinstance(node, ast.Set):
            return {self.eval_node(elt) for elt in node.elts}

        if isinstance(node, ast.Dict):
            return {
                self.eval_node(k): self.eval_node(v)
                for k, v in zip(node.keys, node.values, strict=True)
                if k is not None
            }

        if isinstance(node, ast.Subscript):
            target = self.eval_node(node.value)
            key = self.eval_node(node.slice)
            return target[key]

        raise ExpressionEvaluationError(
            f"Disallowed or unsupported expression element: {type(node).__name__}"
        )


def safe_eval_rule_expression(
    expression: str,
    payload: Mapping[str, Any],
) -> bool:
    """Safely evaluate an axiom rule expression against payload mapping without using eval()."""
    expr_str = expression.strip()
    if not expr_str:
        return True
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as err:
        raise ExpressionEvaluationError(
            f"Invalid syntax in rule expression '{expression}': {err}"
        ) from err

    evaluator = _SafeAstEvaluator(payload)
    try:
        result = evaluator.eval_node(tree)
    except (TypeError, KeyError, IndexError, ZeroDivisionError) as err:
        raise ExpressionEvaluationError(
            f"Evaluation error in rule expression '{expression}': {err}"
        ) from err

    return bool(result)


def _validate_promotion_evidence(
    element_type: str,
    name: str,
    evidence: EvidenceRecord | None,
    min_observations: int,
    min_distinct_sessions: int,
) -> None:
    """Validate that observational evidence meets promotion criteria (P7, P8)."""
    if evidence is None:
        raise PromotionCriteriaNotMetError(
            f"Cannot promote {element_type} '{name}': observation count (0) < threshold ({min_observations})"
        )
    if evidence.observation_count < min_observations:
        raise PromotionCriteriaNotMetError(
            f"Cannot promote {element_type} '{name}': observation count ({evidence.observation_count}) < threshold ({min_observations})"
        )
    if evidence.session_count < min_distinct_sessions:
        raise PromotionCriteriaNotMetError(
            f"Cannot promote {element_type} '{name}': distinct session count ({evidence.session_count}) < floor ({min_distinct_sessions})"
        )
    if len(evidence.contradicting_observations) > 0:
        raise PromotionCriteriaNotMetError(
            f"Cannot promote {element_type} '{name}': open contradictions present: {evidence.contradicting_observations}"
        )


class OntologyEngine:
    """Manages tiered living knowledge graph with deterministic validation and induction."""

    def __init__(
        self,
        agent_id: str = "default",
        namespace_iri: str = "https://uclone-x.ai/ontology/default",
        version: int = 1,
    ) -> None:
        self._agent_id = agent_id
        self._namespace_iri = namespace_iri
        self._version = version
        self._concepts: dict[str, OntologyConcept] = {}
        self._relations: list[OntologyRelation] = []
        self._axioms: dict[str, OntologyAxiom] = {}
        self._contradiction_history: list[dict[str, Any]] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def namespace_iri(self) -> str:
        return self._namespace_iri

    @property
    def version(self) -> int:
        return self._version

    @property
    def content_hash(self) -> str:
        return self.compute_content_hash()

    # ----------------------------------------------------------------------------------
    # 1. Deterministic Content Hash (Asserted Tier Only)
    # ----------------------------------------------------------------------------------

    def compute_content_hash(self) -> str:
        """Compute SHA-256 content hash over canonical representation of ASSERTED terms only."""
        asserted_concepts = [
            c
            for c in self._concepts.values()
            if c.tier
            in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            )
        ]
        asserted_concepts.sort(key=lambda c: c.name)

        asserted_relations = [
            r
            for r in self._relations
            if r.tier
            in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            )
        ]
        asserted_relations.sort(
            key=lambda r: (r.source_entity, r.predicate, r.target_entity, r.is_directed)
        )

        asserted_axioms = [
            a
            for a in self._axioms.values()
            if a.tier
            in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            )
        ]
        asserted_axioms.sort(key=lambda a: a.name)

        canonical_payload: dict[str, Any] = {
            "agent_id": self._agent_id,
            "axioms": [
                {
                    "name": a.name,
                    "object_value": a.object_value,
                    "predicate": a.predicate,
                    "rule_expression": a.rule_expression,
                    "subject_entity": a.subject_entity,
                }
                for a in asserted_axioms
            ],
            "concepts": [
                {
                    "attributes": dict(sorted(c.attributes.items())),
                    "name": c.name,
                    "parent_type": c.parent_type,
                    "required_fields": sorted(c.required_fields),
                }
                for c in asserted_concepts
            ],
            "namespace_iri": self._namespace_iri,
            "relations": [
                {
                    "is_directed": r.is_directed,
                    "predicate": r.predicate,
                    "source_entity": r.source_entity,
                    "target_entity": r.target_entity,
                }
                for r in asserted_relations
            ],
        }

        canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------------------------------
    # 2. Registration & Query Methods (Protocols Compliance)
    # ----------------------------------------------------------------------------------

    def register_entity(self, entity: OntologyConcept) -> None:
        """Register or update an entity concept in the ontology."""
        self._concepts[entity.name] = entity
        if entity.tier in (
            OntologyTier.ASSERTED,
            OntologyTier.ASSERTED_CORE,
            OntologyTier.ASSERTED_DOMAIN,
        ):
            self._version += 1

    def register_relation(self, relation: OntologyRelation) -> None:
        """Register a relationship triplet between domain entities."""
        # Check if identical relation already exists
        for i, existing in enumerate(self._relations):
            if (
                existing.source_entity == relation.source_entity
                and existing.predicate == relation.predicate
                and existing.target_entity == relation.target_entity
                and existing.is_directed == relation.is_directed
            ):
                self._relations[i] = relation
                if relation.tier in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                ):
                    self._version += 1
                return

        self._relations.append(relation)
        if relation.tier in (
            OntologyTier.ASSERTED,
            OntologyTier.ASSERTED_CORE,
            OntologyTier.ASSERTED_DOMAIN,
        ):
            self._version += 1

    def register_axiom(self, axiom: OntologyAxiom) -> None:
        """Register an axiom invariant rule."""
        self._axioms[axiom.name] = axiom
        if axiom.tier in (
            OntologyTier.ASSERTED,
            OntologyTier.ASSERTED_CORE,
            OntologyTier.ASSERTED_DOMAIN,
        ):
            self._version += 1

    def get_entity(self, name: str) -> OntologyConcept | None:
        """Retrieve entity concept definition by name."""
        return self._concepts.get(name)

    def get_concept(self, name: str) -> OntologyConcept | None:
        """Retrieve entity concept definition by name."""
        return self._concepts.get(name)

    def get_axiom(self, name: str) -> OntologyAxiom | None:
        """Retrieve axiom definition by name."""
        return self._axioms.get(name)

    def list_concepts(self, tier: OntologyTier | None = None) -> list[OntologyConcept]:
        """List all concepts, optionally filtered by tier."""
        if tier is None:
            return list(self._concepts.values())
        return [c for c in self._concepts.values() if c.tier == tier]

    def list_relations(self, tier: OntologyTier | None = None) -> list[OntologyRelation]:
        """List all relations, optionally filtered by tier."""
        if tier is None:
            return list(self._relations)
        return [r for r in self._relations if r.tier == tier]

    def list_axioms(self, tier: OntologyTier | None = None) -> list[OntologyAxiom]:
        """List all axioms, optionally filtered by tier."""
        if tier is None:
            return list(self._axioms.values())
        return [a for a in self._axioms.values() if a.tier == tier]

    def get_active_invariants(
        self,
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> list[OntologyAxiom]:
        """Retrieve active invariant rules filtered by domain and ontology tier (P7, P8).

        Parameters
        ----------
        domain:
            Optional domain or subject entity name filter.
        tier_filter:
            - 'asserted' (default): Returns only asserted invariants (tier ASSERTED, ASSERTED_CORE, ASSERTED_DOMAIN, INDUCED_ENFORCING).
            - 'candidate': Returns only candidate invariants (tier INDUCED_CANDIDATE).
            - 'all': Returns invariants across all tiers.
        """
        if tier_filter not in ("asserted", "candidate", "all"):
            raise ValueError(
                f"Invalid tier_filter: '{tier_filter}' (expected 'asserted', 'candidate', or 'all')"
            )

        results: list[OntologyAxiom] = []
        for axiom in self._axioms.values():
            # Tier filtering
            if tier_filter == "asserted":
                if axiom.tier not in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                    OntologyTier.INDUCED_ENFORCING,
                ):
                    continue
            elif tier_filter == "candidate":
                if axiom.tier != OntologyTier.INDUCED_CANDIDATE:
                    continue
            elif tier_filter == "all":
                pass

            # Domain filtering
            if domain is not None:
                matches_domain = (
                    axiom.domain == domain
                    or axiom.subject_entity == domain
                    or (axiom.domain is None and axiom.subject_entity == domain)
                )
                if not matches_domain:
                    continue

            results.append(axiom)

        return results

    # ----------------------------------------------------------------------------------
    # 3. Human Teaching (ASSERTED Tier)
    # ----------------------------------------------------------------------------------

    def teach_concept(
        self,
        name: str,
        description: str = "",
        parent_type: str | None = None,
        attributes: Mapping[str, str] | None = None,
        required_fields: tuple[str, ...] | list[str] = (),
        provenance: Provenance | None = None,
    ) -> OntologyConcept:
        """Explicitly teach an asserted concept (human wins, precedence 100)."""
        attrs = dict(attributes) if attributes else {}
        req = tuple(required_fields)

        concept = OntologyConcept(
            name=name,
            description=description,
            parent_type=parent_type,
            attributes=attrs,
            required_fields=req,
            tier=OntologyTier.ASSERTED,
            precedence=tier_to_precedence(OntologyTier.ASSERTED),
        )

        self._concepts[name] = concept
        self._version += 1
        return concept

    def teach_relation(
        self,
        source_entity: str,
        predicate: str,
        target_entity: str,
        is_directed: bool = True,
        provenance: Provenance | None = None,
    ) -> OntologyRelation:
        """Explicitly teach an asserted relationship triplet."""
        rel = OntologyRelation(
            source_entity=source_entity,
            predicate=predicate,
            target_entity=target_entity,
            is_directed=is_directed,
            tier=OntologyTier.ASSERTED,
            precedence=tier_to_precedence(OntologyTier.ASSERTED),
        )
        self.register_relation(rel)
        return rel

    def teach_axiom(
        self,
        name: str,
        subject_entity: str,
        predicate: str = "",
        object_value: str = "",
        rule_expression: str = "",
        description: str = "",
        domain: str | None = None,
        tier: OntologyTier = OntologyTier.ASSERTED,
        provenance: Provenance | None = None,
    ) -> OntologyAxiom:
        """Explicitly teach an asserted invariant rule axiom."""
        tier_final = normalize_tier(tier)
        axiom = OntologyAxiom(
            name=name,
            subject_entity=subject_entity,
            predicate=predicate,
            object_value=object_value,
            rule_expression=rule_expression,
            description=description,
            domain=domain,
            tier=tier_final,
            precedence=tier_to_precedence(tier_final),
        )
        self.register_axiom(axiom)
        return axiom

    # Aliases for explicit human assertion
    assert_concept = teach_concept
    assert_relation = teach_relation
    assert_axiom = teach_axiom

    def teach_directive(
        self,
        directive: str,
        provenance: Provenance | None = None,
        allow_unparsed_concept: bool = False,
    ) -> OntologyConcept | OntologyRelation | OntologyAxiom:
        """Parse natural language or structured directive into an asserted element (P6 fail-fast)."""
        text = directive.strip()
        if not text:
            if allow_unparsed_concept:
                return self.teach_concept(
                    name=f"concept_{int(time.time())}", description="", provenance=provenance
                )
            raise UnparseableDirectiveError(text=text)

        # ----------------------------------------------------------------------
        # 1. Concept definitions
        # ----------------------------------------------------------------------
        # Form 1a: [define] concept|entity <Name> [extends|is a <Parent>] [with attributes <k:v, ...>] [requires <f1, f2>]
        concept_match = re.match(
            r"^(?:define\s+)?(?:concept|entity)\s+([a-zA-Z0-9_-]+)(?:\s+(?:extends|is\s+an?)\s+([a-zA-Z0-9_-]+))?(?:\s+with\s+attributes\s+([^\[\]]+?))?(?:\s+requires\s+(.+))?$",
            text,
            re.IGNORECASE,
        )
        if concept_match:
            name = concept_match.group(1)
            parent = concept_match.group(2)
            attrs_str = concept_match.group(3)
            reqs_str = concept_match.group(4)

            attrs: dict[str, str] = {}
            if attrs_str:
                for pair in attrs_str.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        attrs[k.strip()] = v.strip()
                    elif pair.strip():
                        attrs[pair.strip()] = "str"

            reqs: list[str] = []
            if reqs_str:
                reqs = [r.strip() for r in reqs_str.split(",") if r.strip()]

            return self.teach_concept(
                name=name,
                parent_type=parent,
                attributes=attrs,
                required_fields=reqs,
                description=f"Asserted concept {name}",
                provenance=provenance,
            )

        # Form 1b: "<X> is a <Y>" or "<X> is an <Y>" (e.g. "User is a Person")
        is_a_match = re.match(
            r"^([a-zA-Z0-9_-]+)\s+is\s+an?\s+([a-zA-Z0-9_-]+)$",
            text,
            re.IGNORECASE,
        )
        if is_a_match:
            name = is_a_match.group(1)
            parent = is_a_match.group(2)
            return self.teach_concept(
                name=name,
                parent_type=parent,
                description=f"{name} is a {parent}",
                provenance=provenance,
            )

        # Form 1c: "define concept <X> [: <description>]"
        def_concept_match = re.match(
            r"^define\s+(?:concept|entity)\s+([a-zA-Z0-9_-]+)(?:\s*:\s*(.+))?$",
            text,
            re.IGNORECASE,
        )
        if def_concept_match:
            name = def_concept_match.group(1)
            desc = (def_concept_match.group(2) or f"Asserted concept {name}").strip()
            return self.teach_concept(
                name=name,
                description=desc,
                provenance=provenance,
            )

        # ----------------------------------------------------------------------
        # 2. Relation triplets
        # ----------------------------------------------------------------------
        # Form 2: [relation: ] <Source> -> <predicate> -> <Target> (also --> or =>)
        rel_match = re.match(
            r"^(?:relation:?\s+)?([a-zA-Z0-9_-]+)\s*(?:->|-->|=>)\s*([a-zA-Z0-9_-]+)\s*(?:->|-->|=>)\s*([a-zA-Z0-9_-]+)$",
            text,
            re.IGNORECASE,
        )
        if rel_match:
            return self.teach_relation(
                source_entity=rel_match.group(1),
                predicate=rel_match.group(2),
                target_entity=rel_match.group(3),
                provenance=provenance,
            )

        # ----------------------------------------------------------------------
        # 3. Axioms, Invariants & Structured Constraint Rules
        # ----------------------------------------------------------------------
        # Form 3a: (rule|axiom|invariant) <Name>: <subject> [requires] <predicate> == <value>
        axiom_eq_match = re.match(
            r"^(?:rule|axiom|invariant)\s+([a-zA-Z0-9_-]+):\s*([a-zA-Z0-9_-]+)\s+(?:requires\s+)?([a-zA-Z0-9_-]+)\s*==\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if axiom_eq_match:
            name = axiom_eq_match.group(1)
            subject = axiom_eq_match.group(2)
            predicate = axiom_eq_match.group(3)
            val = axiom_eq_match.group(4).strip()
            return self.teach_axiom(
                name=name,
                subject_entity=subject,
                predicate=predicate,
                object_value=val,
                rule_expression=f"{predicate} == {val}",
                description=text,
                provenance=provenance,
            )

        # Form 3b: "all <X> must|shall|should <Y>"
        # e.g., "all internal gRPC calls must use mTLS encryption"
        all_must_match = re.match(
            r"^all\s+(.+?)\s+(?:must|shall|should)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if all_must_match:
            subject_raw = all_must_match.group(1).strip()
            constraint_raw = all_must_match.group(2).strip()

            # Parse verb and object if applicable (e.g. "use mTLS encryption")
            verb_match = re.match(
                r"^(use|have|be|contain|include|require|satisfy|implement|follow)\s+(.+)$",
                constraint_raw,
                re.IGNORECASE,
            )
            if verb_match:
                predicate = verb_match.group(1).lower()
                object_val = verb_match.group(2).strip()
            else:
                predicate = "must"
                object_val = constraint_raw

            rule_name = _slugify_axiom_name(f"all_{subject_raw}_must_{constraint_raw}")
            return self.teach_axiom(
                name=rule_name,
                subject_entity=subject_raw,
                predicate=predicate,
                object_value=object_val,
                rule_expression=f"all {subject_raw} must {constraint_raw}",
                description=text,
                provenance=provenance,
            )

        # Form 3c: "whenever <X>, <Y>" or "whenever <X> then <Y>"
        # e.g., "whenever token expires, refresh token"
        whenever_match = re.match(
            r"^whenever\s+(.+?)(?:,\s*|\s+then\s+)(.+)$",
            text,
            re.IGNORECASE,
        )
        if whenever_match:
            subject_raw = whenever_match.group(1).strip()
            action_raw = whenever_match.group(2).strip()
            rule_name = _slugify_axiom_name(f"whenever_{subject_raw}_{action_raw}")
            return self.teach_axiom(
                name=rule_name,
                subject_entity=subject_raw,
                predicate="triggers",
                object_value=action_raw,
                rule_expression=f"whenever {subject_raw}, then {action_raw}",
                description=text,
                provenance=provenance,
            )

        # Form 3d: (rule|axiom|invariant) <Name>: <expression>
        named_rule_match = re.match(
            r"^(?:rule|axiom|invariant)\s+([a-zA-Z0-9_-]+):\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if named_rule_match:
            name = named_rule_match.group(1)
            body = named_rule_match.group(2).strip()
            return self.teach_axiom(
                name=name,
                subject_entity=name,
                rule_expression=body,
                description=text,
                provenance=provenance,
            )

        # Form 3e: "<X> requires <Y>" or "<X> require <Y>"
        # e.g., "Deployment requires replicas", "User requires email"
        requires_match = re.match(
            r"^([a-zA-Z0-9_ -]+?)\s+requires?\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if requires_match:
            subject_raw = requires_match.group(1).strip()
            req_raw = requires_match.group(2).strip()
            rule_name = _slugify_axiom_name(f"{subject_raw}_requires_{req_raw}")
            return self.teach_axiom(
                name=rule_name,
                subject_entity=subject_raw,
                predicate="requires",
                object_value=req_raw,
                rule_expression=f"{subject_raw} requires {req_raw}",
                description=text,
                provenance=provenance,
            )

        # Form 3f: "<X> must <Y>"
        # e.g., "PaymentService must use TLS"
        must_match = re.match(
            r"^([a-zA-Z0-9_ -]+?)\s+must\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if must_match:
            subject_raw = must_match.group(1).strip()
            action_raw = must_match.group(2).strip()
            rule_name = _slugify_axiom_name(f"{subject_raw}_must_{action_raw}")
            return self.teach_axiom(
                name=rule_name,
                subject_entity=subject_raw,
                predicate="must",
                object_value=action_raw,
                rule_expression=f"{subject_raw} must {action_raw}",
                description=text,
                provenance=provenance,
            )

        # ----------------------------------------------------------------------
        # 4. Fallback / Unparseable Directive (P6 Fail-Fast)
        # ----------------------------------------------------------------------
        if allow_unparsed_concept:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", text[:40]).strip("_")
            if not safe_name:
                safe_name = f"concept_{int(time.time())}"
            return self.teach_concept(name=safe_name, description=text, provenance=provenance)

        raise UnparseableDirectiveError(text=text)

    # ----------------------------------------------------------------------------------
    # 4. Autonomous Induction & Contradiction Resolution
    # ----------------------------------------------------------------------------------

    def induce_concept(
        self,
        name: str,
        description: str = "",
        parent_type: str | None = None,
        attributes: Mapping[str, str] | None = None,
        required_fields: tuple[str, ...] | list[str] = (),
        source_session: str | None = None,
        model_id: str | None = None,
        confidence: float = 0.9,
    ) -> OntologyConcept:
        """Induce a concept observation from agent execution turns."""
        attrs = dict(attributes) if attributes else {}
        req = tuple(required_fields)
        now_ts = utc_now_iso()

        existing = self._concepts.get(name)

        if existing is not None:
            # Check contradiction with ASSERTED concept
            if existing.tier == OntologyTier.ASSERTED:
                # Check for attribute type conflict or parent conflict
                has_conflict = False
                reasons: list[str] = []
                for k, v in attrs.items():
                    if k in existing.attributes and existing.attributes[k] != v:
                        has_conflict = True
                        reasons.append(
                            f"Attribute '{k}' type conflict: asserted '{existing.attributes[k]}', induced '{v}'"
                        )
                if parent_type and existing.parent_type and parent_type != existing.parent_type:
                    has_conflict = True
                    reasons.append(
                        f"Parent type conflict: asserted '{existing.parent_type}', induced '{parent_type}'"
                    )

                if has_conflict:
                    # Asserted wins unconditionally; record rejection
                    self._contradiction_history.append(
                        {
                            "concept": name,
                            "tier": OntologyTier.ASSERTED,
                            "reasons": reasons,
                            "session": source_session,
                            "timestamp": now_ts,
                        }
                    )
                    raise OntologyContradictionError(
                        f"Induced concept '{name}' contradicts asserted definition: {'; '.join(reasons)}"
                    )
                return existing

            # Check contradiction with INDUCED_ENFORCING concept -> Immediate Demotion!
            if existing.tier == OntologyTier.INDUCED_ENFORCING:
                has_conflict = False
                reasons: list[str] = []
                for k, v in attrs.items():
                    if k in existing.attributes and existing.attributes[k] != v:
                        has_conflict = True
                        reasons.append(
                            f"Attribute '{k}' type conflict: enforcing '{existing.attributes[k]}', new '{v}'"
                        )
                if parent_type and existing.parent_type and parent_type != existing.parent_type:
                    has_conflict = True
                    reasons.append(
                        f"Parent type conflict: enforcing '{existing.parent_type}', new '{parent_type}'"
                    )

                if has_conflict:
                    # Immediate demotion to candidate
                    old_evidence = existing.evidence or EvidenceRecord(
                        observation_count=1, first_seen=now_ts, last_seen=now_ts
                    )
                    demoted_evidence = old_evidence.model_copy(
                        update={
                            "contradicting_observations": old_evidence.contradicting_observations
                            + tuple(reasons),
                            "last_seen": now_ts,
                        }
                    )
                    demoted = existing.model_copy(
                        update={
                            "tier": OntologyTier.INDUCED_CANDIDATE,
                            "precedence": tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
                            "evidence": demoted_evidence,
                        }
                    )
                    self._concepts[name] = demoted
                    self._contradiction_history.append(
                        {
                            "concept": name,
                            "tier": OntologyTier.INDUCED_ENFORCING,
                            "action": "demoted_to_candidate",
                            "reasons": reasons,
                            "session": source_session,
                            "timestamp": now_ts,
                        }
                    )
                    return demoted

            # Updating existing INDUCED_CANDIDATE
            evidence = existing.evidence or EvidenceRecord(
                observation_count=0, first_seen=now_ts, last_seen=now_ts
            )
            sessions = list(evidence.originating_sessions)
            obs_count = evidence.observation_count + 1
            if source_session and source_session not in sessions:
                sessions.append(source_session)

            merged_attrs = dict(existing.attributes)
            merged_attrs.update(attrs)
            merged_reqs = tuple(sorted(set(existing.required_fields).union(req)))

            updated_evidence = evidence.model_copy(
                update={
                    "observation_count": obs_count,
                    "last_seen": now_ts,
                    "originating_sessions": tuple(sessions),
                    "source_session": source_session or evidence.source_session,
                    "model_id": model_id or evidence.model_id,
                    "confidence": max(evidence.confidence, confidence),
                }
            )

            updated = existing.model_copy(
                update={
                    "attributes": merged_attrs,
                    "required_fields": merged_reqs,
                    "evidence": updated_evidence,
                }
            )
            self._concepts[name] = updated
            return updated

        # Brand new candidate
        sessions_tuple = (source_session,) if source_session else ()
        new_evidence = EvidenceRecord(
            observation_count=1,
            first_seen=now_ts,
            last_seen=now_ts,
            source_session=source_session,
            originating_sessions=sessions_tuple,
            model_id=model_id,
            confidence=confidence,
        )

        concept = OntologyConcept(
            name=name,
            description=description,
            parent_type=parent_type,
            attributes=attrs,
            required_fields=req,
            tier=OntologyTier.INDUCED_CANDIDATE,
            confidence=confidence,
            evidence=new_evidence,
            precedence=tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
        )
        self._concepts[name] = concept
        return concept

    def induce_relation(
        self,
        source_entity: str,
        predicate: str,
        target_entity: str,
        is_directed: bool = True,
        source_session: str | None = None,
        model_id: str | None = None,
        confidence: float = 0.9,
    ) -> OntologyRelation:
        """Induce a relationship observation from agent turns."""
        now_ts = utc_now_iso()
        for i, existing in enumerate(self._relations):
            if (
                existing.source_entity == source_entity
                and existing.predicate == predicate
                and existing.target_entity == target_entity
            ):
                if existing.tier in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                ):
                    return existing
                evidence = existing.evidence or EvidenceRecord(
                    observation_count=0, first_seen=now_ts, last_seen=now_ts
                )
                sessions = list(evidence.originating_sessions)
                obs_count = evidence.observation_count + 1
                if source_session and source_session not in sessions:
                    sessions.append(source_session)

                updated_evidence = evidence.model_copy(
                    update={
                        "observation_count": obs_count,
                        "last_seen": now_ts,
                        "originating_sessions": tuple(sessions),
                        "source_session": source_session or evidence.source_session,
                        "model_id": model_id or evidence.model_id,
                        "confidence": max(evidence.confidence, confidence),
                    }
                )
                updated = existing.model_copy(update={"evidence": updated_evidence})
                self._relations[i] = updated
                return updated

        sessions_tuple = (source_session,) if source_session else ()
        new_evidence = EvidenceRecord(
            observation_count=1,
            first_seen=now_ts,
            last_seen=now_ts,
            source_session=source_session,
            originating_sessions=sessions_tuple,
            model_id=model_id,
            confidence=confidence,
        )
        rel = OntologyRelation(
            source_entity=source_entity,
            predicate=predicate,
            target_entity=target_entity,
            is_directed=is_directed,
            tier=OntologyTier.INDUCED_CANDIDATE,
            confidence=confidence,
            evidence=new_evidence,
            precedence=tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
        )
        self._relations.append(rel)
        return rel

    def induce_axiom(
        self,
        name: str,
        subject_entity: str,
        predicate: str = "",
        object_value: str = "",
        rule_expression: str = "",
        description: str = "",
        domain: str | None = None,
        source_session: str | None = None,
        model_id: str | None = None,
        confidence: float = 0.85,
    ) -> OntologyAxiom:
        """Induce a candidate invariant rule axiom from agent execution turns (P7)."""
        now_ts = utc_now_iso()
        if name in self._axioms:
            existing = self._axioms[name]
            if existing.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            ):
                return existing

            # Check if enforcing axiom is contradicted
            if existing.tier == OntologyTier.INDUCED_ENFORCING:
                has_conflict = False
                reasons: list[str] = []
                if object_value and existing.object_value and object_value != existing.object_value:
                    has_conflict = True
                    reasons.append(
                        f"Axiom '{name}' object conflict: enforcing '{existing.object_value}', new '{object_value}'"
                    )
                if (
                    rule_expression
                    and existing.rule_expression
                    and rule_expression != existing.rule_expression
                ):
                    has_conflict = True
                    reasons.append(
                        f"Axiom '{name}' rule conflict: enforcing '{existing.rule_expression}', new '{rule_expression}'"
                    )
                if has_conflict:
                    old_evidence = existing.evidence or EvidenceRecord(
                        observation_count=1, first_seen=now_ts, last_seen=now_ts
                    )
                    demoted_evidence = old_evidence.model_copy(
                        update={
                            "contradicting_observations": old_evidence.contradicting_observations
                            + tuple(reasons),
                            "last_seen": now_ts,
                        }
                    )
                    demoted = existing.model_copy(
                        update={
                            "tier": OntologyTier.INDUCED_CANDIDATE,
                            "precedence": tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
                            "evidence": demoted_evidence,
                        }
                    )
                    self._axioms[name] = demoted
                    self._contradiction_history.append(
                        {
                            "axiom": name,
                            "tier": OntologyTier.INDUCED_ENFORCING,
                            "action": "demoted_to_candidate",
                            "reasons": reasons,
                            "session": source_session,
                            "timestamp": now_ts,
                        }
                    )
                    return demoted

            # Updating existing candidate
            evidence = existing.evidence or EvidenceRecord(
                observation_count=0, first_seen=now_ts, last_seen=now_ts
            )
            sessions = list(evidence.originating_sessions)
            obs_count = evidence.observation_count + 1
            if source_session and source_session not in sessions:
                sessions.append(source_session)

            updated_evidence = evidence.model_copy(
                update={
                    "observation_count": obs_count,
                    "last_seen": now_ts,
                    "originating_sessions": tuple(sessions),
                    "source_session": source_session or evidence.source_session,
                    "model_id": model_id or evidence.model_id,
                    "confidence": max(evidence.confidence, confidence),
                }
            )
            updated_ax = existing.model_copy(
                update={
                    "predicate": predicate or existing.predicate,
                    "object_value": object_value or existing.object_value,
                    "rule_expression": rule_expression or existing.rule_expression,
                    "description": description or existing.description,
                    "domain": domain or existing.domain,
                    "evidence": updated_evidence,
                }
            )
            self._axioms[name] = updated_ax
            return updated_ax

        # Brand new candidate
        sessions_tuple = (source_session,) if source_session else ()
        new_evidence = EvidenceRecord(
            observation_count=1,
            first_seen=now_ts,
            last_seen=now_ts,
            source_session=source_session,
            originating_sessions=sessions_tuple,
            model_id=model_id,
            confidence=confidence,
        )
        axiom = OntologyAxiom(
            name=name,
            subject_entity=subject_entity,
            predicate=predicate,
            object_value=object_value,
            rule_expression=rule_expression,
            description=description,
            domain=domain,
            tier=OntologyTier.INDUCED_CANDIDATE,
            confidence=confidence,
            evidence=new_evidence,
            precedence=tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
        )
        self._axioms[name] = axiom
        return axiom

    # ----------------------------------------------------------------------------------
    # 5. Promotion, Demotion & Retraction (Forget)
    # ----------------------------------------------------------------------------------

    def promote(
        self,
        name: str,
        approver: str = "human:developer",
        force: bool = False,
        min_observations: int = 5,
        min_distinct_sessions: int = 2,
        min_sessions: int | None = None,
        target_tier: OntologyTier = OntologyTier.INDUCED_ENFORCING,
    ) -> OntologyConcept | OntologyRelation | OntologyAxiom:
        """Promote an induced candidate to induced-enforcing or asserted tier (P7, P8)."""
        target_tier_final = normalize_tier(target_tier)
        effective_min_sessions = min_sessions if min_sessions is not None else min_distinct_sessions

        # 1. Check concepts
        if name in self._concepts:
            concept = self._concepts[name]
            if concept.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            ):
                raise OntologyPromotionError(f"Concept '{name}' is already asserted.")
            if concept.tier == target_tier_final and not force:
                return concept

            if not force:
                _validate_promotion_evidence(
                    element_type="concept",
                    name=name,
                    evidence=concept.evidence,
                    min_observations=min_observations,
                    min_distinct_sessions=effective_min_sessions,
                )

            promoted = concept.model_copy(
                update={
                    "tier": target_tier_final,
                    "precedence": tier_to_precedence(target_tier_final),
                }
            )
            self._concepts[name] = promoted
            self._version += 1
            return promoted

        # 2. Check axioms
        if name in self._axioms:
            axiom = self._axioms[name]
            if axiom.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            ):
                raise OntologyPromotionError(f"Axiom '{name}' is already asserted.")
            if axiom.tier == target_tier_final and not force:
                return axiom

            if not force:
                _validate_promotion_evidence(
                    element_type="axiom",
                    name=name,
                    evidence=axiom.evidence,
                    min_observations=min_observations,
                    min_distinct_sessions=effective_min_sessions,
                )

            promoted_ax = axiom.model_copy(
                update={
                    "tier": target_tier_final,
                    "precedence": tier_to_precedence(target_tier_final),
                }
            )
            self._axioms[name] = promoted_ax
            self._version += 1
            return promoted_ax

        # 3. Check relations
        for i, relation in enumerate(self._relations):
            rel_keys = (
                f"{relation.source_entity}:{relation.predicate}:{relation.target_entity}",
                f"{relation.source_entity}->{relation.predicate}->{relation.target_entity}",
                f"{relation.source_entity} -> {relation.predicate} -> {relation.target_entity}",
                f"relation:({relation.source_entity}->{relation.predicate}->{relation.target_entity})",
                relation.predicate,
            )
            if name in rel_keys:
                if relation.tier in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                ):
                    raise OntologyPromotionError(f"Relation '{name}' is already asserted.")
                if relation.tier == target_tier_final and not force:
                    return relation

                if not force:
                    _validate_promotion_evidence(
                        element_type="relation",
                        name=name,
                        evidence=relation.evidence,
                        min_observations=min_observations,
                        min_distinct_sessions=effective_min_sessions,
                    )

                promoted_rel = relation.model_copy(
                    update={
                        "tier": target_tier_final,
                        "precedence": tier_to_precedence(target_tier_final),
                    }
                )
                self._relations[i] = promoted_rel
                self._version += 1
                return promoted_rel

        raise OntologyPromotionError(f"Ontology element '{name}' not found for promotion.")

    def demote(
        self,
        name: str,
        reason: str = "Demoted by reviewer",
    ) -> OntologyConcept | OntologyRelation | OntologyAxiom:
        """Demote an induced-enforcing element to candidate staging."""
        now_ts = utc_now_iso()

        if name in self._concepts:
            concept = self._concepts[name]
            if concept.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            ):
                raise OntologyViolationError(
                    f"Cannot demote asserted concept '{name}': asserted terms are human-governed."
                )
            old_evidence = concept.evidence or EvidenceRecord(
                observation_count=1, first_seen=now_ts, last_seen=now_ts
            )
            updated_evidence = old_evidence.model_copy(
                update={
                    "contradicting_observations": old_evidence.contradicting_observations
                    + (reason,),
                    "last_seen": now_ts,
                }
            )
            demoted = concept.model_copy(
                update={
                    "tier": OntologyTier.INDUCED_CANDIDATE,
                    "precedence": tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
                    "evidence": updated_evidence,
                }
            )
            self._concepts[name] = demoted
            self._version += 1
            return demoted

        if name in self._axioms:
            axiom = self._axioms[name]
            if axiom.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            ):
                raise OntologyViolationError(
                    f"Cannot demote asserted axiom '{name}': asserted terms are human-governed."
                )
            old_evidence = axiom.evidence or EvidenceRecord(
                observation_count=1, first_seen=now_ts, last_seen=now_ts
            )
            updated_evidence = old_evidence.model_copy(
                update={
                    "contradicting_observations": old_evidence.contradicting_observations
                    + (reason,),
                    "last_seen": now_ts,
                }
            )
            demoted_ax = axiom.model_copy(
                update={
                    "tier": OntologyTier.INDUCED_CANDIDATE,
                    "precedence": tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
                    "evidence": updated_evidence,
                }
            )
            self._axioms[name] = demoted_ax
            self._version += 1
            return demoted_ax

        for i, relation in enumerate(self._relations):
            rel_keys = (
                f"{relation.source_entity}:{relation.predicate}:{relation.target_entity}",
                f"{relation.source_entity}->{relation.predicate}->{relation.target_entity}",
                f"{relation.source_entity} -> {relation.predicate} -> {relation.target_entity}",
                f"relation:({relation.source_entity}->{relation.predicate}->{relation.target_entity})",
                relation.predicate,
            )
            if name in rel_keys:
                if relation.tier in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                ):
                    raise OntologyViolationError(
                        f"Cannot demote asserted relation '{name}': asserted terms are human-governed."
                    )
                old_evidence = relation.evidence or EvidenceRecord(
                    observation_count=1, first_seen=now_ts, last_seen=now_ts
                )
                updated_evidence = old_evidence.model_copy(
                    update={
                        "contradicting_observations": old_evidence.contradicting_observations
                        + (reason,),
                        "last_seen": now_ts,
                    }
                )
                demoted_rel = relation.model_copy(
                    update={
                        "tier": OntologyTier.INDUCED_CANDIDATE,
                        "precedence": tier_to_precedence(OntologyTier.INDUCED_CANDIDATE),
                        "evidence": updated_evidence,
                    }
                )
                self._relations[i] = demoted_rel
                self._version += 1
                return demoted_rel

        raise OntologyViolationError(f"Ontology element '{name}' not found for demotion.")

    def forget(self, name: str, force: bool = False) -> list[str]:
        """Retract an ontology term, checking dependencies to prevent orphaned invariants."""
        retracted: list[str] = []

        if name in self._concepts:
            # Check dependent asserted elements
            asserted_dependents: list[str] = []

            # 1. Child concepts
            for c_name, child in self._concepts.items():
                if (
                    child.parent_type == name
                    and c_name != name
                    and child.tier == OntologyTier.ASSERTED
                ):
                    asserted_dependents.append(f"child_concept:{c_name}")

            # 2. Relations
            for r in self._relations:
                if (
                    r.source_entity == name or r.target_entity == name
                ) and r.tier == OntologyTier.ASSERTED:
                    rel_repr = f"relation:({r.source_entity}->{r.predicate}->{r.target_entity})"
                    asserted_dependents.append(rel_repr)

            # 3. Axioms
            for a_name, a in self._axioms.items():
                if a.subject_entity == name and a.tier == OntologyTier.ASSERTED:
                    asserted_dependents.append(f"axiom:{a_name}")

            if asserted_dependents and not force:
                raise OntologyRetractionBlockedError(
                    f"Cannot forget '{name}': blocking asserted dependents exist: {', '.join(asserted_dependents)}"
                )

            # Retract concept
            del self._concepts[name]
            retracted.append(name)

            # Cascade delete child concepts and axioms
            for c_name, child in list(self._concepts.items()):
                if child.parent_type == name:
                    del self._concepts[c_name]
                    retracted.append(c_name)

            for a_name, a in list(self._axioms.items()):
                if a.subject_entity == name:
                    del self._axioms[a_name]
                    retracted.append(a_name)

            # Clean relations
            self._relations = [
                r for r in self._relations if r.source_entity != name and r.target_entity != name
            ]

            self._version += 1
            return retracted

        if name in self._axioms:
            del self._axioms[name]
            retracted.append(name)
            self._version += 1
            return retracted

        raise OntologyViolationError(f"Ontology element '{name}' not found for retraction.")

    # ----------------------------------------------------------------------------------
    # 6. Tier-1 High-Speed Deterministic Turn Validation
    # ----------------------------------------------------------------------------------

    def validate_entity(
        self,
        entity_name: str,
        data: Mapping[str, Any] | dict[str, Any],
        pinned_content_hash: str | None = None,
    ) -> ValidationResult:
        """Validate an input/output payload against active ontology schema."""
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        # Content hash verification
        if pinned_content_hash is not None:
            current_hash = self.compute_content_hash()
            if current_hash != pinned_content_hash:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                return ValidationResult(
                    is_valid=False,
                    errors=(
                        f"Content hash mismatch: pinned '{pinned_content_hash}', active '{current_hash}'",
                    ),
                    content_hash=current_hash,
                    latency_ms=latency_ms,
                    provenance=Provenance.primary("ontology_validator"),
                )

        concept = self._concepts.get(entity_name)

        if concept is None:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ValidationResult(
                is_valid=False,
                errors=(f"Entity '{entity_name}' is not defined in the ontology.",),
                latency_ms=latency_ms,
                provenance=Provenance.primary("ontology_validator"),
            )

        # Candidate tier is advisory only, never enforcing
        if concept.tier == OntologyTier.INDUCED_CANDIDATE:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ValidationResult(
                is_valid=True,
                warnings=(
                    f"Entity '{entity_name}' is induced-candidate (advisory only, not enforcing).",
                ),
                matched_tier=OntologyTier.INDUCED_CANDIDATE,
                content_hash=concept.content_hash,
                latency_ms=latency_ms,
                provenance=Provenance.primary("ontology_validator"),
            )

        # Enforcing tiers: ASSERTED, ASSERTED_CORE, ASSERTED_DOMAIN, INDUCED_ENFORCING
        # Collect concept ancestor hierarchy (from entity_name up parent_type chain)
        ancestor_concepts: list[OntologyConcept] = []
        curr_name: str | None = entity_name
        visited_concepts: set[str] = set()
        while curr_name is not None and curr_name not in visited_concepts:
            visited_concepts.add(curr_name)
            curr_c = self._concepts.get(curr_name)
            if curr_c is None:
                break
            if curr_c.tier != OntologyTier.INDUCED_CANDIDATE:
                ancestor_concepts.append(curr_c)
            curr_name = curr_c.parent_type

        # 1. Required fields check across concept and ancestor hierarchy
        seen_required: set[str] = set()
        for c in ancestor_concepts:
            for req_field in c.required_fields:
                if req_field not in seen_required:
                    seen_required.add(req_field)
                    if req_field not in data or data[req_field] is None:
                        errors.append(
                            f"Missing required field '{req_field}' for entity '{entity_name}'."
                        )

        # 2. Attribute type validation across concept and ancestor hierarchy
        merged_attributes: dict[str, str] = {}
        for c in reversed(ancestor_concepts):
            merged_attributes.update(c.attributes)

        for attr_name, expected_type in merged_attributes.items():
            if attr_name in data and data[attr_name] is not None:
                val: Any = data[attr_name]
                t_clean = expected_type.lower().strip()
                val_type_name = val.__class__.__name__
                if t_clean in ("str", "string") and not isinstance(val, str):
                    errors.append(f"Attribute '{attr_name}' expected str, got {val_type_name}.")
                elif t_clean in ("int", "integer") and (
                    not isinstance(val, int) or isinstance(val, bool)
                ):
                    errors.append(f"Attribute '{attr_name}' expected int, got {val_type_name}.")
                elif t_clean in ("float", "number") and not isinstance(val, (int, float)):
                    errors.append(f"Attribute '{attr_name}' expected float, got {val_type_name}.")
                elif t_clean in ("bool", "boolean") and not isinstance(val, bool):
                    errors.append(f"Attribute '{attr_name}' expected bool, got {val_type_name}.")
                elif t_clean in ("list", "array") and not isinstance(val, (list, tuple)):
                    errors.append(f"Attribute '{attr_name}' expected list, got {val_type_name}.")
                elif t_clean in ("dict", "object") and not isinstance(val, dict):
                    errors.append(f"Attribute '{attr_name}' expected dict, got {val_type_name}.")

        # 3. Axioms validation (attached to entity_name or any of its ancestors)
        ancestor_names: set[str] = {c.name for c in ancestor_concepts}
        for axiom in self._axioms.values():
            if axiom.subject_entity in ancestor_names and axiom.tier in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
                OntologyTier.INDUCED_ENFORCING,
            ):
                # 3a. Simple predicate-object equality check
                if axiom.predicate and axiom.object_value:
                    if axiom.predicate in data:
                        actual_val = str(data[axiom.predicate])
                        if actual_val != axiom.object_value:
                            errors.append(
                                f"Axiom '{axiom.name}' violated: '{axiom.predicate}' must equal '{axiom.object_value}', got '{actual_val}'."
                            )

                # 3b. AST rule expression evaluation
                if axiom.rule_expression and axiom.rule_expression.strip():
                    try:
                        passed = safe_eval_rule_expression(axiom.rule_expression, data)
                        if not passed:
                            errors.append(
                                f"Axiom '{axiom.name}' violated: rule expression '{axiom.rule_expression}' evaluated to False for entity '{entity_name}'."
                            )
                    except Exception as exc:
                        errors.append(
                            f"Axiom '{axiom.name}' violated: rule expression '{axiom.rule_expression}' evaluation error: {exc}"
                        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=tuple(errors),
            warnings=tuple(warnings),
            matched_tier=concept.tier,
            content_hash=concept.content_hash,
            latency_ms=latency_ms,
            provenance=Provenance.primary("ontology_validator"),
        )

    # ----------------------------------------------------------------------------------
    # 7. Serialization & LinkML YAML
    # ----------------------------------------------------------------------------------

    def export_graph(self) -> dict[str, Any]:
        """Export all active concepts, axioms, relations, and hierarchy nodes in JSON-serializable structure."""
        concepts_data: list[dict[str, Any]] = [
            {
                "name": c.name,
                "parent_type": c.parent_type,
                "tier": c.tier.value if hasattr(c.tier, "value") else str(c.tier),
                "precedence": c.precedence,
                "attributes": dict(c.attributes),
                "required_fields": list(c.required_fields),
                "content_hash": c.content_hash,
                "description": c.description,
                "confidence": c.confidence,
                "iri": c.iri,
                "provenance": {
                    "source": (
                        c.evidence.source_session
                        if c.evidence and c.evidence.source_session
                        else "linkml_core"
                    ),
                    "origin": (
                        "axiomatic"
                        if c.tier
                        in (
                            OntologyTier.ASSERTED,
                            OntologyTier.ASSERTED_CORE,
                            OntologyTier.ASSERTED_DOMAIN,
                        )
                        else "derived"
                    ),
                    "immutable": c.tier
                    in (
                        OntologyTier.ASSERTED,
                        OntologyTier.ASSERTED_CORE,
                        OntologyTier.ASSERTED_DOMAIN,
                    ),
                },
            }
            for c in self._concepts.values()
        ]

        relations_data: list[dict[str, Any]] = [
            {
                "id": f"rel_{i + 1}",
                "source": r.source_entity,
                "predicate": r.predicate,
                "target": r.target_entity,
                "is_directed": r.is_directed,
                "tier": r.tier.value if hasattr(r.tier, "value") else str(r.tier),
                "confidence": r.confidence,
                "content_hash": r.content_hash,
                "precedence": r.precedence,
            }
            for i, r in enumerate(self._relations)
        ]

        axioms_data: list[dict[str, Any]] = [
            {
                "name": a.name,
                "subject_entity": a.subject_entity,
                "predicate": a.predicate,
                "object_value": a.object_value,
                "rule_expression": a.rule_expression,
                "description": a.description,
                "domain": a.domain,
                "tier": a.tier.value if hasattr(a.tier, "value") else str(a.tier),
                "confidence": a.confidence,
                "content_hash": a.content_hash,
                "precedence": a.precedence,
            }
            for a in self._axioms.values()
        ]

        asserted_count = sum(
            1
            for c in self._concepts.values()
            if c.tier
            in (
                OntologyTier.ASSERTED,
                OntologyTier.ASSERTED_CORE,
                OntologyTier.ASSERTED_DOMAIN,
            )
        )
        induced_enforcing_count = sum(
            1 for c in self._concepts.values() if c.tier == OntologyTier.INDUCED_ENFORCING
        )
        induced_candidate_count = sum(
            1 for c in self._concepts.values() if c.tier == OntologyTier.INDUCED_CANDIDATE
        )

        return {
            "concepts": concepts_data,
            "relations": relations_data,
            "axioms": axioms_data,
            "total_concepts": len(self._concepts),
            "total_axioms": len(self._axioms),
            "total_relations": len(self._relations),
            "summary": {
                "total_concepts": len(self._concepts),
                "total_relations": len(self._relations),
                "total_axioms": len(self._axioms),
                "asserted_count": asserted_count,
                "induced_enforcing_count": induced_enforcing_count,
                "induced_candidate_count": induced_candidate_count,
            },
        }

    def export_linkml_yaml(self) -> str:
        """Serialize active ontology to standard LinkML YAML specification."""
        classes_dict: dict[str, Any] = {}
        slots_dict: dict[str, Any] = {}

        for concept in self._concepts.values():
            c_info: dict[str, Any] = {
                "description": concept.description or f"Domain entity {concept.name}",
            }
            if concept.parent_type:
                c_info["is_a"] = concept.parent_type
            if concept.attributes:
                c_info["slots"] = list(concept.attributes.keys())
                for slot_name, slot_type in concept.attributes.items():
                    slots_dict[slot_name] = {
                        "range": slot_type,
                        "required": slot_name in concept.required_fields,
                    }
            if concept.required_fields:
                c_info["required"] = list(concept.required_fields)
            classes_dict[concept.name] = c_info

        payload: dict[str, Any] = {
            "id": self._namespace_iri,
            "name": f"{self._agent_id}_ontology",
            "version": self._version,
            "content_hash": self.compute_content_hash(),
            "prefixes": {
                "linkml": "https://w3id.org/linkml/",
                "default": f"{self._namespace_iri}/",
            },
            "default_prefix": "default",
            "classes": classes_dict,
            "slots": slots_dict,
        }

        return yaml.dump(payload, sort_keys=False)

    def save_to_yaml(self, file_path: Path) -> None:
        """Save ontology definition and metadata to YAML file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dict: dict[str, Any] = {
            "agent_id": self._agent_id,
            "namespace_iri": self._namespace_iri,
            "version": self._version,
            "content_hash": self.compute_content_hash(),
            "concepts": [c.model_dump(mode="json") for c in self._concepts.values()],
            "relations": [r.model_dump(mode="json") for r in self._relations],
            "axioms": [a.model_dump(mode="json") for a in self._axioms.values()],
        }
        export_dict: dict[str, Any] = json.loads(json.dumps(raw_dict, default=str))
        file_path.write_text(yaml.dump(export_dict, sort_keys=False), encoding="utf-8")

    def load_from_yaml(self, file_path: Path) -> None:
        """Load ontology from a YAML file."""
        if not file_path.is_file():
            return
        content = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return

        data_dict = cast(dict[str, Any], data)
        if "agent_id" in data_dict:
            self._agent_id = str(data_dict["agent_id"])
        if "namespace_iri" in data_dict:
            self._namespace_iri = str(data_dict["namespace_iri"])
        if "version" in data_dict and isinstance(data_dict["version"], int):
            self._version = data_dict["version"]

        self._concepts.clear()
        concepts_list = data_dict.get("concepts", [])
        if isinstance(concepts_list, list):
            for c_dict in cast(list[Any], concepts_list):
                if isinstance(c_dict, dict):
                    concept = OntologyConcept.model_validate(c_dict)
                    self._concepts[concept.name] = concept

        self._relations.clear()
        relations_list = data_dict.get("relations", [])
        if isinstance(relations_list, list):
            for r_dict in cast(list[Any], relations_list):
                if isinstance(r_dict, dict):
                    relation = OntologyRelation.model_validate(r_dict)
                    self._relations.append(relation)

        self._axioms.clear()
        axioms_list = data_dict.get("axioms", [])
        if isinstance(axioms_list, list):
            for a_dict in cast(list[Any], axioms_list):
                if isinstance(a_dict, dict):
                    axiom = OntologyAxiom.model_validate(a_dict)
                    self._axioms[axiom.name] = axiom


# Alias for OntologyEngine
OntologyService = OntologyEngine


class OntologyInducer:
    """Autonomous entity and relation extraction from turns."""

    def __init__(self, engine: OntologyEngine | None = None) -> None:
        self._engine = engine or OntologyEngine()

    @property
    def engine(self) -> OntologyEngine:
        return self._engine

    async def induce_from_turn(
        self,
        turn_text: str,
        tool_results: tuple[ToolResult, ...],
    ) -> tuple[OntologyConcept, ...]:
        """Extract candidate entities and constraints from successful turns."""
        candidates: list[OntologyConcept] = []

        pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_]{2,}\b")
        seen: set[str] = set()
        for match in pattern.finditer(turn_text):
            w = match.group(0)
            if w not in seen and len(w) > 3:
                seen.add(w)
                concept = self._engine.induce_concept(
                    name=w,
                    description="Candidate entity extracted from turn context",
                    confidence=0.85,
                )
                candidates.append(concept)

        return tuple(candidates)


class SessionKnowledgeExtractor:
    """Autonomous knowledge induction and rule extraction from dialogue sessions (P7)."""

    def __init__(self, engine: OntologyEngine | None = None) -> None:
        self._engine = engine or OntologyEngine()

    @property
    def engine(self) -> OntologyEngine:
        return self._engine

    def extract_from_session(
        self,
        messages: list[ChatMessage] | tuple[ChatMessage, ...],
        session_id: str = "session",
        model_id: str = "uclone_x",
    ) -> tuple[OntologyConcept, ...]:
        """Extract domain concepts and invariants from session turns and stage as Candidate tier."""
        candidates: list[OntologyConcept] = []
        seen: set[str] = set()

        for msg in messages:
            content = msg.content or ""
            # Extract capitalized domain concepts
            pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_]{2,}\b")
            for match in pattern.finditer(content):
                name = match.group(0)
                if name not in seen and len(name) > 3:
                    seen.add(name)
                    # Check if already exists in engine
                    existing = self._engine.get_concept(name)
                    if existing is None:
                        c = self._engine.induce_concept(
                            name=name,
                            description=f"Auto-induced domain concept from session turn ({msg.role.value})",
                            confidence=0.85,
                            source_session=session_id,
                            model_id=model_id,
                        )
                        candidates.append(c)

        return tuple(candidates)

    def inject_rules_to_prompt(
        self,
        base_system_prompt: str = "",
        domain: str | None = None,
        tier_filter: Literal["asserted", "candidate", "all"] = "asserted",
    ) -> str:
        """Inject active Asserted and Enforcing ontology concepts/axioms into system prompt (P7, P8)."""
        active_axioms = self._engine.get_active_invariants(domain=domain, tier_filter=tier_filter)
        if tier_filter == "asserted":
            active_concepts = [
                c
                for c in self._engine.list_concepts()
                if c.tier
                in (
                    OntologyTier.ASSERTED,
                    OntologyTier.ASSERTED_CORE,
                    OntologyTier.ASSERTED_DOMAIN,
                    OntologyTier.INDUCED_ENFORCING,
                )
            ]
        elif tier_filter == "candidate":
            active_concepts = [
                c for c in self._engine.list_concepts() if c.tier == OntologyTier.INDUCED_CANDIDATE
            ]
        else:
            active_concepts = self._engine.list_concepts()

        if not active_axioms and not active_concepts:
            return base_system_prompt

        lines = [base_system_prompt.strip(), "\n\n[Active Domain Ontology Invariants]:"]
        for c in active_concepts:
            lines.append(f"- Concept: {c.name} ({c.description or 'No description'})")
        for a in active_axioms:
            lines.append(f"- Rule ({a.tier.value}): {a.name} -> {a.rule_expression or a.predicate}")

        return "\n".join(lines)
