"""Justification-carrying value types for the ontology entailment closure (issue #138).

The point of this module is not that facts can be derived; it is that a derived fact can
say **why** it holds. A closure that entails without justifying is an unauditable black
box: an operator reading it cannot tell a consequence of a human-asserted axiom from a
consequence of a bad guess. So the closure stores, for every derived fact, the rule that
fired and the premises it rested on, and `Closure.explain` walks that record down to
asserted leaves.

Three invariants are enforced here rather than left to convention:

* **Fact identity is the triple.** `Fact.id` is a SHA-256 over the canonical JSON of
  `(subject, predicate, object)`, recomputed on every construction and never accepted from
  the caller. Justifications reference ids, so the same triple must yield the same id in
  another process, in another run, on another machine.
* **A derived fact without a justification is rejected**, not tolerated. `Closure`
  raises `OntologyViolationError` on construction rather than serving a fact whose
  provenance was lost, which is the Principle 6 shape applied to entailment.
* **`explain` is well-founded.** Derivation graphs cycle (symmetry alone guarantees it),
  so `explain` traverses only proof steps that are *well-founded*: every premise must be
  derivable from asserted leaves by a route that does not pass through the conclusion
  itself. Well-foundedness is a reachability property, not a length one -- a support that
  happens to run through deeper premises is still a support, and `explain` reports it.
  What is excluded is exactly circular support (A justifies B justifies A), which is not
  a proof of anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, model_validator

from uclone_x.errors import OntologyViolationError
from uclone_x.ontology.models import OntologyTier, normalize_tier

__all__ = [
    "DISJOINT_WITH",
    "DOMAIN",
    "FUNCTIONAL_PROPERTY",
    "HORN_RULE",
    "INVERSE_FUNCTIONAL_PROPERTY",
    "INVERSE_OF",
    "MAX_CARDINALITY",
    "MIN_CARDINALITY",
    "RANGE",
    "SUBCLASS_OF",
    "SUBPROPERTY_OF",
    "SYMMETRIC_PROPERTY",
    "TRANSITIVE_PROPERTY",
    "TYPE_PREDICATE",
    "Closure",
    "Fact",
    "Inconsistency",
    "ProofStep",
    "UnsupportedAxiom",
    "canonical_predicate",
    "compute_fact_id",
]

TYPE_PREDICATE: Final = "type"
"""Predicate used for class membership assertions, i.e. `type(individual, Class)`."""

SUBCLASS_OF: Final = "subClassOf"
SUBPROPERTY_OF: Final = "subPropertyOf"
DOMAIN: Final = "domain"
RANGE: Final = "range"
INVERSE_OF: Final = "inverseOf"
TRANSITIVE_PROPERTY: Final = "transitiveProperty"
SYMMETRIC_PROPERTY: Final = "symmetricProperty"
FUNCTIONAL_PROPERTY: Final = "functionalProperty"
INVERSE_FUNCTIONAL_PROPERTY: Final = "inverseFunctionalProperty"
DISJOINT_WITH: Final = "disjointWith"
MAX_CARDINALITY: Final = "maxCardinality"
MIN_CARDINALITY: Final = "minCardinality"

HORN_RULE: Final = "hornRule"
"""Predicate of a safe Horn rule fact: `hornRule(<rule name>, <canonical rule text>)`.

A rule is a fact so that a firing can cite it. `ProofStep.premises` holds `Fact.id` values,
so a rule that is not a fact cannot appear in its own justification -- and a derivation
that cannot name the rule that produced it is not an explanation.
"""

_PREDICATE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "rdf:type": TYPE_PREDICATE,
        "rdfs:subclassof": SUBCLASS_OF,
        "rdfs:subpropertyof": SUBPROPERTY_OF,
        "rdfs:domain": DOMAIN,
        "rdfs:range": RANGE,
        "owl:inverseof": INVERSE_OF,
        "owl:transitiveproperty": TRANSITIVE_PROPERTY,
        "owl:symmetricproperty": SYMMETRIC_PROPERTY,
        "owl:functionalproperty": FUNCTIONAL_PROPERTY,
        "owl:inversefunctionalproperty": INVERSE_FUNCTIONAL_PROPERTY,
        "owl:disjointwith": DISJOINT_WITH,
        "owl:maxcardinality": MAX_CARDINALITY,
        "owl:mincardinality": MIN_CARDINALITY,
    }
)
"""Namespaced spellings of the built-in vocabulary, folded onto one canonical name.

This is the *identity* half of issue #138: `rdf:type` and `type` are one predicate, and
two agents writing them differently must not produce two facts that never meet. Folding is
deliberately narrow -- only these prefixed forms of the built-in vocabulary -- because
guessing that two *domain* predicates mean the same thing is exactly the drift the
ontology exists to prevent.
"""


def canonical_predicate(predicate: str) -> str:
    """Fold a namespaced spelling of the built-in vocabulary onto its canonical name."""
    stripped = predicate.strip()
    return _PREDICATE_ALIASES.get(stripped.lower(), stripped)


def compute_fact_id(subject: str, predicate: str, object_value: str) -> str:
    """Compute the canonical deterministic SHA-256 identity of a triple.

    The digest is taken over sorted, separator-normalised JSON so that the same triple
    yields the same identity across processes and runs. The predicate is canonicalised
    first, so `rdf:type(x, C)` and `type(x, C)` are one fact rather than two. Nothing but
    the three triple components contributes: tier and derivation status are properties of
    *how* a fact entered the closure, not of *which* fact it is.
    """
    canonical: dict[str, str] = {
        "object": object_value,
        "predicate": canonical_predicate(predicate),
        "subject": subject,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Fact(BaseModel):
    """One subject-predicate-object assertion held in the closure.

    `derived` is the audit-relevant flag: `False` marks an asserted leaf -- something a
    human or an upstream observation put in -- and `True` marks a consequence the
    reasoner computed, which therefore must carry at least one `ProofStep`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = ""
    subject: str
    predicate: str
    object: str
    tier: OntologyTier = OntologyTier.ASSERTED
    derived: bool = False

    @model_validator(mode="before")
    @classmethod
    def _populate_fact_defaults(cls, data: Any) -> Any:
        """Canonicalise the predicate and tier, and recompute `id`, ignoring any supplied id."""
        if not isinstance(data, dict):
            return data
        d: dict[str, Any] = dict(cast(dict[str, Any], data))
        d["tier"] = normalize_tier(d["tier"]) if "tier" in d else OntologyTier.ASSERTED
        if "predicate" in d:
            d["predicate"] = canonical_predicate(str(d["predicate"]))
        d["id"] = compute_fact_id(
            str(d.get("subject", "")),
            str(d.get("predicate", "")),
            str(d.get("object", "")),
        )
        return d

    @property
    def triple(self) -> tuple[str, str, str]:
        """Return the identity-bearing `(subject, predicate, object)` components."""
        return (self.subject, self.predicate, self.object)


class ProofStep(BaseModel):
    """A single rule application: these premises, by this rule, yield this conclusion."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rule: str
    premises: tuple[str, ...]
    conclusion: str


class Inconsistency(BaseModel):
    """A detected contradiction, naming the facts that cannot hold together.

    `facts` lists the conflicting data facts in sorted order, followed by the schema fact
    that licensed the constraint, so a reader can see both the conflict and the rule it
    violates. `explanation` is one sentence written for a human who must act on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: str
    facts: tuple[str, ...]
    explanation: str


class UnsupportedAxiom(BaseModel):
    """A schema statement the reasoner refused to act on, reported rather than dropped.

    Silently ignoring an axiom a caller believed was in force is the defect Principle 6
    forbids: the caller cannot distinguish "this constraint held" from "this constraint
    was never evaluated". Every refusal is therefore recorded on the closure with the
    reason, in band.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: str
    tier: OntologyTier
    reason: str


class Closure:
    """A materialised entailment closure and the justification record behind it."""

    __slots__ = ("_depths", "_facts", "_grounded_without", "_justifications", "_unsupported_axioms")

    def __init__(
        self,
        facts: Mapping[str, Fact],
        justifications: Mapping[str, tuple[ProofStep, ...]] | None = None,
        unsupported_axioms: tuple[UnsupportedAxiom, ...] = (),
    ) -> None:
        """Build a closure, rejecting any derived fact whose justification is missing."""
        self._facts: Mapping[str, Fact] = MappingProxyType(dict(facts))
        recorded: dict[str, tuple[ProofStep, ...]] = {
            fact_id: tuple(steps)
            for fact_id, steps in (justifications or {}).items()
            if len(steps) > 0
        }
        self._justifications: Mapping[str, tuple[ProofStep, ...]] = MappingProxyType(recorded)
        self._unsupported_axioms: tuple[UnsupportedAxiom, ...] = tuple(unsupported_axioms)
        self._validate()
        self._depths: Mapping[str, int] = MappingProxyType(self._compute_depths())
        self._grounded_without: dict[str, frozenset[str]] = {}

    @property
    def facts(self) -> Mapping[str, Fact]:
        """Every fact in the closure, asserted and derived, keyed by `Fact.id`."""
        return self._facts

    @property
    def justifications(self) -> Mapping[str, tuple[ProofStep, ...]]:
        """Every recorded proof step, keyed by the `Fact.id` it concludes.

        This is the raw record, including circular justifications that `explain` omits
        because they prove nothing. Alternative *well-founded* supports are not filtered
        out here and are not filtered out by `explain` either: "what else would have
        produced this" is exactly the question an impact analysis asks, and an answer that
        named only one of several live supports would mislead the operator who acts on it.
        """
        return self._justifications

    @property
    def unsupported_axioms(self) -> tuple[UnsupportedAxiom, ...]:
        """Schema statements that were refused, with the reason each was refused."""
        return self._unsupported_axioms

    def is_entailed(self, subject: str, predicate: str, object: str) -> bool:
        """Return whether the given triple is present in the closure, asserted or derived."""
        return compute_fact_id(subject, predicate, object) in self._facts

    def get(self, subject: str, predicate: str, object: str) -> Fact | None:
        """Return the fact for the given triple, or `None` if it is not entailed."""
        return self._facts.get(compute_fact_id(subject, predicate, object))

    def explain(self, fact_id: str) -> tuple[ProofStep, ...]:
        """Return every well-founded proof step supporting a fact, asserted-resting last.

        An asserted fact returns an empty tuple: it rests on nothing, so there is nothing
        to explain.

        **What is returned.** Every step reachable from `fact_id` that is *well-founded*:
        each of its premises must be derivable from asserted leaves by some route that
        does not pass through the step's own conclusion. Nothing is dropped for being a
        longer route than another. If four independent supports keep `fact_id` alive, all
        four are here -- which is the property an operator needs before concluding that
        revoking one named fact retracts the conclusion.

        **What is omitted, exactly.** Only circular steps: those with a premise whose sole
        support runs back through the conclusion (A justifies B justifies A). Such a step
        is not a proof, and both halves of the cycle stay visible on `justifications`.
        Note also that this returns a *set of steps*, not an enumeration of proof trees --
        a step shared by two supports appears once.

        **Ordering.** By decreasing derivation depth of the conclusion; within one depth,
        the queried fact's own steps come first, then steps resting on at least one derived
        premise, then steps resting only on asserted leaves. So the last step always
        bottoms out on asserted facts.

        **Termination.** Guaranteed by the traversal, not by the filter: each fact id is
        expanded at most once (`visited`), and the closure is finite. Admissibility itself
        is a monotone fixed point over that same finite set.

        Raises:
            KeyError: if `fact_id` is not in the closure. A miss is not answered with an
                empty tuple, which would be indistinguishable from an asserted fact.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            raise KeyError(f"fact id '{fact_id}' is not in this closure")
        if not fact.derived:
            return ()

        collected: dict[tuple[str, tuple[str, ...], str], ProofStep] = {}
        visited: set[str] = set()
        pending: list[str] = [fact_id]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if not self._facts[current].derived:
                continue
            for step in self._well_founded_steps(current):
                collected[(step.rule, step.premises, step.conclusion)] = step
                pending.extend(step.premises)

        return tuple(sorted(collected.values(), key=lambda step: self._ordering_key(fact_id, step)))

    def depth(self, fact_id: str) -> int:
        """Return the derivation depth of a fact: 0 for asserted, else shortest proof height.

        Depth orders `explain` output and answers "how far is this from an assertion". It
        is deliberately *not* an admissibility criterion: a support is judged by whether it
        is grounded, never by how long it is.
        """
        return self._depths.get(fact_id, 0)

    def _ordering_key(
        self, queried: str, step: ProofStep
    ) -> tuple[int, int, int, str, str, tuple[str, ...]]:
        """Return the sort key implementing the ordering contract documented on `explain`."""
        rests_on_leaves = all(not self._facts[premise].derived for premise in step.premises)
        return (
            -self._depths.get(step.conclusion, 0),
            0 if step.conclusion == queried else 1,
            1 if rests_on_leaves else 0,
            step.conclusion,
            step.rule,
            step.premises,
        )

    def _well_founded_steps(self, fact_id: str) -> tuple[ProofStep, ...]:
        """Return the steps concluding `fact_id` whose premises are all grounded without it.

        "Grounded without it" is the whole rule: a premise must be derivable from asserted
        leaves along a route that never uses `fact_id`. That is what stops a fact from
        being offered as its own reason. Proof length plays no part.
        """
        grounded = self._grounded_excluding(fact_id)
        return tuple(
            step
            for step in self._justifications.get(fact_id, ())
            if all(premise in grounded for premise in step.premises)
        )

    def _grounded_excluding(self, excluded: str) -> frozenset[str]:
        """Return the facts derivable from asserted leaves without ever using `excluded`.

        A least fixed point over a finite fact set: asserted facts other than `excluded`
        seed it, and a derived fact joins once some justification has all its premises
        already in. The set only grows and is bounded by the closure, so the loop ends.
        Results are memoised per excluded id because `explain` asks the same question once
        per fact it walks.
        """
        cached = self._grounded_without.get(excluded)
        if cached is not None:
            return cached
        grounded: set[str] = {
            fact_id
            for fact_id, fact in self._facts.items()
            if not fact.derived and fact_id != excluded
        }
        changed = True
        while changed:
            changed = False
            for fact_id, fact in self._facts.items():
                if fact_id == excluded or fact_id in grounded or not fact.derived:
                    continue
                for step in self._justifications.get(fact_id, ()):
                    if all(premise in grounded for premise in step.premises):
                        grounded.add(fact_id)
                        changed = True
                        break
        frozen = frozenset(grounded)
        self._grounded_without[excluded] = frozen
        return frozen

    def _validate(self) -> None:
        """Enforce the closure invariants that make `explain` meaningful."""
        for fact_id, steps in self._justifications.items():
            if fact_id not in self._facts:
                raise OntologyViolationError(
                    f"justification recorded for unknown fact id '{fact_id}'"
                )
            for step in steps:
                if step.conclusion != fact_id:
                    raise OntologyViolationError(
                        f"proof step for '{fact_id}' concludes '{step.conclusion}' instead"
                    )
                for premise in step.premises:
                    if premise not in self._facts:
                        raise OntologyViolationError(
                            f"rule '{step.rule}' cites premise '{premise}', which is not in the closure"
                        )
        for fact_id, fact in self._facts.items():
            if fact.derived and fact_id not in self._justifications:
                raise OntologyViolationError(
                    f"derived fact {fact.triple} carries no justification; "
                    "an unexplainable derivation is a defect, not a result"
                )

    def _compute_depths(self) -> dict[str, int]:
        """Compute the shortest well-founded proof height of every fact.

        Asserted facts have depth 0. A derived fact's depth is one more than the deepest
        premise of its shallowest justification. A fact supported only cyclically never
        acquires a depth, and `depth` then reports 0 for it -- but nothing is admitted or
        rejected on the strength of a depth: this map only orders `explain` output.
        """
        depths: dict[str, int] = {
            fact_id: 0 for fact_id, fact in self._facts.items() if not fact.derived
        }
        changed = True
        while changed:
            changed = False
            for fact_id, fact in self._facts.items():
                if not fact.derived:
                    continue
                best: int | None = None
                for step in self._justifications.get(fact_id, ()):
                    premise_depths = [depths.get(premise) for premise in step.premises]
                    if any(depth is None for depth in premise_depths):
                        continue
                    deepest = max(
                        (depth for depth in premise_depths if depth is not None), default=0
                    )
                    candidate = deepest + 1
                    if best is None or candidate < best:
                        best = candidate
                if best is not None and best < depths.get(fact_id, best + 1):
                    depths[fact_id] = best
                    changed = True
        return depths
