"""Forward-chaining materialisation and consistency checking (issue #138, E1 slice).

`materialize` runs the nine derivation rules of `rules.py` to a fixed point, recording a
`ProofStep` for every firing, and returns a `Closure` that can answer *why*.
`check_consistency` runs the three detectors over a materialised closure and returns typed
`Inconsistency` records naming the facts that cannot hold together.

**Termination.** Each round rebuilds the schema index and fires every rule over the whole
store. Three quantities change and all three are monotone and bounded: the fact set grows
within a finite vocabulary of subjects, predicates and objects; the recorded proof-step set
grows within rules x premise-tuples over that fact set; and a fact's tier only ever
strengthens, across three levels. A round that changes none of the three ends the loop, so
a fact that is already present cannot re-fire indefinitely. Transitivity combined with
symmetry cycles the *justification* graph, not the loop.

The Horn rules of the thirteenth capability preserve this. The grammar has no function
symbols and every rule is *safe* -- each head variable is bound by the body -- so a firing
can only conclude over constants that already occur in the closure or literally in the
rule. No rule can invent a term, the vocabulary therefore stays finite, and the bound above
still holds. This is Datalog, and the fixed point is reached.

**Tier propagation.** A conclusion is no stronger than its weakest premise. Because the
schema premises are asserted by construction (see `rules.AxiomIndex`), only the data
premises can weaken a conclusion -- an induced observation yields an induced consequence,
and nothing induced is ever laundered into an asserted fact.

**Entailment and consistency are two questions, and a caller must ask both.** This is the
contract, stated rather than left to be discovered: `Closure.is_entailed` reports whether a
triple is *present in the closure*, and nothing more. It does not consult
`check_consistency`, and it is unchanged by a contradiction elsewhere in the graph. So
`is_entailed(skill, type, Approved)` can return `True` for a skill the very same closure
knows to be contradictory -- both an approval and a rejection, say -- and a caller that
asks only the first question will act on that `True`.

It is deliberately this way, for three reasons. Membership is a dictionary lookup and must
stay one; folding detection into it would make a cheap call quietly expensive and run the
detectors once per query. Under classical entailment a contradictory theory entails
everything, so suppressing one answer would be arbitrary rather than principled. And,
decisively, *which* fact to retract in the face of a contradiction is a policy question the
reasoner cannot answer -- returning `False` would claim it had resolved a conflict it had
merely found, which is a worse failure than reporting the conflict plainly.

The obligation this places on callers is therefore explicit: **any decision that grants
authority -- admission, approval, capability -- must call `check_consistency` and treat a
non-empty result as disqualifying, in addition to checking entailment.** Entailment alone
answers "does this follow?", not "is the graph it follows from coherent?".
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any, Final

from uclone_x.errors import OntologyViolationError
from uclone_x.ontology.justification import (
    Closure,
    Fact,
    Inconsistency,
    ProofStep,
    UnsupportedAxiom,
)
from uclone_x.ontology.models import OntologyAxiom, OntologyTier, tier_to_precedence
from uclone_x.ontology.rules import (
    DERIVATION_RULES,
    DETECTION_RULES,
    AxiomIndex,
    Derivation,
    FactView,
    interpret_axioms,
)

__all__ = ["MAX_ROUNDS", "OntologyReasoner", "check_consistency", "materialize"]

MAX_ROUNDS: Final = 1000
"""Round ceiling. Termination is argued structurally; this only turns a hypothetical
runaway into a loud failure instead of a hang, per Principle 6."""


def materialize(facts: Iterable[Fact], axioms: Iterable[OntologyAxiom]) -> Closure:
    """Compute the entailment closure of `facts` under the asserted subset of `axioms`.

    Interpretable asserted axioms are seeded into the closure as schema facts, so that a
    proof step can cite them by `Fact.id`. Every input fact is treated as an asserted leaf:
    `derived` is forced to `False` on the way in, because a caller cannot hand the reasoner
    a derivation it did not perform and expect it to be explainable.

    Refused axioms -- induced tier, unknown predicate, malformed arguments -- derive
    nothing. They, and any `rule_expression` that went unevaluated, are reported on
    `Closure.unsupported_axioms`, so a caller can always tell what was not applied.
    """
    schema_facts, refusals = interpret_axioms(axioms)
    store: dict[str, Fact] = {}
    for incoming in chain(facts, schema_facts):
        seed = incoming if not incoming.derived else incoming.model_copy(update={"derived": False})
        existing = store.get(seed.id)
        if existing is None or _stronger(seed.tier, existing.tier):
            store[seed.id] = seed

    justifications: dict[str, set[ProofStep]] = {}
    index = AxiomIndex(store.values())
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:  # pragma: no cover - structurally unreachable
            raise OntologyViolationError(
                f"forward chaining did not reach a fixed point within {MAX_ROUNDS} rounds"
            )
        index = AxiomIndex(store.values())
        view = FactView(dict(store))
        changed = False
        for rule in DERIVATION_RULES:
            for derivation in rule(index, view):
                if _apply(store, justifications, derivation):
                    changed = True
        if not changed:
            break

    unsupported = _dedupe_refusals(chain(refusals, index.refusals))
    return Closure(
        facts=store,
        justifications={fact_id: _order_steps(steps) for fact_id, steps in justifications.items()},
        unsupported_axioms=unsupported,
    )


def check_consistency(closure: Closure) -> tuple[Inconsistency, ...]:
    """Report every contradiction the three detectors find in a materialised closure.

    The result is deduplicated and ordered by `(kind, facts)`, so two runs over the same
    closure report the same contradictions in the same order.

    This is the second of the two questions described in the module docstring, and nothing
    else asks it. `Closure.is_entailed` does not call this function and is not affected by
    what it finds, so a triple can be entailed by a closure this function reports as
    contradictory. A caller deciding anything security-relevant must ask both and treat a
    non-empty result here as disqualifying regardless of what entailment said.
    """
    index = AxiomIndex(closure.facts.values())
    view = FactView(closure.facts)
    found: set[Inconsistency] = set()
    for detector in DETECTION_RULES:
        found.update(detector(index, view))
    return tuple(sorted(found, key=lambda item: (item.kind, item.facts)))


def _apply(
    store: dict[str, Fact],
    justifications: dict[str, set[ProofStep]],
    derivation: Derivation,
) -> bool:
    """Record one rule firing, returning whether it changed the closure.

    A firing changes the closure when it introduces a new fact, strengthens an existing
    fact's tier, or records a justification not already held. Re-deriving a fact that is
    already present with an already-recorded justification changes nothing, which is what
    stops transitivity and symmetry from cycling forever.
    """
    conclusion = Fact(
        subject=derivation.subject,
        predicate=derivation.predicate,
        object=derivation.object,
        tier=_weakest_tier(store, derivation.premises),
        derived=True,
    )
    changed = False
    existing = store.get(conclusion.id)
    if existing is None:
        store[conclusion.id] = conclusion
        changed = True
    elif _stronger(conclusion.tier, existing.tier):
        store[conclusion.id] = existing.model_copy(update={"tier": conclusion.tier})
        changed = True

    step = ProofStep(
        rule=derivation.rule,
        premises=derivation.premises,
        conclusion=conclusion.id,
    )
    recorded = justifications.setdefault(conclusion.id, set())
    if step not in recorded:
        recorded.add(step)
        changed = True
    return changed


def _weakest_tier(store: dict[str, Fact], premises: tuple[str, ...]) -> OntologyTier:
    """Return the least authoritative tier among the premises: a chain is as weak as its links."""
    weakest = OntologyTier.ASSERTED
    for premise in premises:
        fact = store.get(premise)
        if fact is not None and not _stronger(fact.tier, weakest):
            weakest = fact.tier
    return weakest


def _stronger(candidate: OntologyTier, incumbent: OntologyTier) -> bool:
    """Return whether `candidate` carries strictly more authority than `incumbent`."""
    return tier_to_precedence(candidate) > tier_to_precedence(incumbent)


def _order_steps(steps: set[ProofStep]) -> tuple[ProofStep, ...]:
    """Return proof steps in a stable order so that two runs produce identical records."""
    return tuple(sorted(steps, key=lambda step: (step.rule, step.premises)))


def _dedupe_refusals(refusals: Iterable[UnsupportedAxiom]) -> tuple[UnsupportedAxiom, ...]:
    """Collapse duplicate refusals, keeping a stable order for reproducible reports."""
    unique = {(item.source, item.tier, item.reason): item for item in refusals}
    return tuple(sorted(unique.values(), key=lambda item: (item.source, item.reason)))


class OntologyReasoner:
    """Stateful forward-chaining reasoner and consistency validator for ontology graphs (P7).

    Provides a high-level, stateful interface over `materialize` and `check_consistency`.
    Axioms and seed facts are maintained internally, with the computed entailment `Closure`
    cached lazily until new axioms or facts are added.
    """

    def __init__(
        self,
        axioms: Iterable[OntologyAxiom] = (),
        facts: Iterable[Fact] = (),
    ) -> None:
        self._axioms: list[OntologyAxiom] = list(axioms)
        self._facts: list[Fact] = list(facts)
        self._closure: Closure | None = None

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        facts: Iterable[Fact] = (),
    ) -> OntologyReasoner:
        """Instantiate an OntologyReasoner from an OntologyEngine's registered axioms."""
        engine_axioms: list[OntologyAxiom] = []
        if hasattr(engine, "list_axioms"):
            engine_axioms = list(engine.list_axioms())
        elif hasattr(engine, "get_axioms"):
            engine_axioms = list(engine.get_axioms())
        return cls(axioms=engine_axioms, facts=facts)

    @property
    def axioms(self) -> tuple[OntologyAxiom, ...]:
        """Return all axioms held by this reasoner."""
        return tuple(self._axioms)

    @property
    def facts(self) -> tuple[Fact, ...]:
        """Return all seed facts held by this reasoner."""
        return tuple(self._facts)

    @property
    def closure(self) -> Closure:
        """Return the materialised entailment closure, computing it if not cached."""
        return self.materialize()

    def add_axiom(self, axiom: OntologyAxiom) -> None:
        """Add an axiom and invalidate the cached closure."""
        self._axioms.append(axiom)
        self._closure = None

    def add_axioms(self, axioms: Iterable[OntologyAxiom]) -> None:
        """Add multiple axioms and invalidate the cached closure."""
        self._axioms.extend(axioms)
        self._closure = None

    def add_fact(self, fact: Fact) -> None:
        """Add a seed fact and invalidate the cached closure."""
        self._facts.append(fact)
        self._closure = None

    def add_facts(self, facts: Iterable[Fact]) -> None:
        """Add multiple seed facts and invalidate the cached closure."""
        self._facts.extend(facts)
        self._closure = None

    def clear(self) -> None:
        """Clear all held axioms, seed facts, and the cached closure."""
        self._axioms.clear()
        self._facts.clear()
        self._closure = None

    def materialize(self) -> Closure:
        """Compute or return the cached entailment closure."""
        if self._closure is None:
            self._closure = materialize(self._facts, self._axioms)
        return self._closure

    def check_consistency(self) -> tuple[Inconsistency, ...]:
        """Run contradiction detectors over the materialised closure."""
        closure = self.materialize()
        return check_consistency(closure)

    def validate(self) -> tuple[Inconsistency, ...]:
        """Validate consistency of the graph (alias for check_consistency)."""
        return self.check_consistency()

    def is_entailed(self, subject: str, predicate: str, object_value: str) -> bool:
        """Query whether the triple is present in the entailment closure."""
        closure = self.materialize()
        return closure.is_entailed(subject, predicate, object_value)

    def lookup(self, subject: str, predicate: str, object_value: str) -> Fact | None:
        """Look up an exact fact by (subject, predicate, object)."""
        closure = self.materialize()
        return closure.get(subject, predicate, object_value)

    def lookup_subject(self, subject: str) -> tuple[Fact, ...]:
        """Look up all facts where the subject matches."""
        closure = self.materialize()
        return tuple(f for f in closure.facts.values() if f.subject == subject)
