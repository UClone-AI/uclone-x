"""Unit tests for the justification-recording entailment reasoner (issue #138).

The tests are organised around the claim the reasoner makes: every derived fact can say
why it holds, no induced axiom may contribute, and a contradiction names the facts that
collide. Each of the nine derivation rules is tested twice -- once for firing, once for
staying silent on a near-miss -- because a rule that fires when it should not is the same
defect as one that never fires, only harder to notice.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uclone_x.errors import OntologyViolationError
from uclone_x.ontology.justification import (
    HORN_RULE,
    TYPE_PREDICATE,
    Closure,
    Fact,
    ProofStep,
    canonical_predicate,
    compute_fact_id,
)
from uclone_x.ontology.models import OntologyAxiom, OntologyTier
from uclone_x.ontology.reasoner import check_consistency, materialize
from uclone_x.ontology.rules import (
    KIND_CARDINALITY,
    KIND_DISJOINT,
    KIND_FUNCTIONAL,
    KIND_INVERSE_FUNCTIONAL,
    RULE_DOMAIN_TYPE,
    RULE_HORN,
    RULE_PROPERTY_SYMMETRY,
    RULE_SUBCLASS_TRANSITIVITY,
    RULE_TYPE_PROPAGATION,
    SCHEMA_PREDICATES,
    SUBCLASS_OF,
    SUBPROPERTY_OF,
    HornTerm,
    canonical_horn_text,
    normalize_axiom_kind,
    parse_horn_rule,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def axiom(
    name: str,
    subject_entity: str,
    predicate: str,
    object_value: str = "",
    tier: OntologyTier = OntologyTier.ASSERTED,
    rule_expression: str = "",
) -> OntologyAxiom:
    """Build an axiom with the structured fields the reasoner interprets."""
    return OntologyAxiom(
        name=name,
        subject_entity=subject_entity,
        predicate=predicate,
        object_value=object_value,
        rule_expression=rule_expression,
        tier=tier,
    )


def triple(
    subject: str,
    predicate: str,
    object_value: str,
    tier: OntologyTier = OntologyTier.ASSERTED,
    derived: bool = False,
) -> Fact:
    """Build a fact from a triple."""
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_value,
        tier=tier,
        derived=derived,
    )


def type_of(
    individual: str,
    class_name: str,
    tier: OntologyTier = OntologyTier.ASSERTED,
) -> Fact:
    """Build a class-membership fact."""
    return triple(individual, TYPE_PREDICATE, class_name, tier=tier)


def require(closure: Closure, subject: str, predicate: str, object_value: str) -> Fact:
    """Return an entailed fact, failing the test if it is absent."""
    found = closure.get(subject, predicate, object_value)
    assert found is not None, f"expected ({subject}, {predicate}, {object_value}) to be entailed"
    return found


# --------------------------------------------------------------------------------------
# Fact identity
# --------------------------------------------------------------------------------------


def test_fact_id_is_deterministic_across_constructions() -> None:
    first = triple("Task#7", TYPE_PREDICATE, "Task")
    second = triple("Task#7", TYPE_PREDICATE, "Task", tier=OntologyTier.INDUCED_CANDIDATE)
    assert first.id == second.id
    assert first.id == compute_fact_id("Task#7", TYPE_PREDICATE, "Task")
    # A literal digest pins determinism across processes and runs, not merely within one.
    assert first.id == "abfb71ea915073c36a23a77a25d45f042066fda4c5f4bfcecdea7e686e260fb4"
    assert first.triple == ("Task#7", TYPE_PREDICATE, "Task")


def test_fact_id_ignores_a_caller_supplied_value_and_separates_triples() -> None:
    forged = Fact(id="not-a-hash", subject="a", predicate="p", object="b")
    assert forged.id == compute_fact_id("a", "p", "b")
    assert forged.id != triple("a", "p", "c").id
    assert forged.id != triple("b", "p", "a").id


def test_namespaced_spellings_of_the_builtin_vocabulary_fold_onto_one_fact() -> None:
    """`rdf:type` and `type` are one predicate; two agents spelling it differently must meet."""
    assert canonical_predicate("rdf:type") == TYPE_PREDICATE
    assert canonical_predicate("RDFS:subClassOf") == SUBCLASS_OF
    assert canonical_predicate("hasVerdict") == "hasVerdict"

    prefixed = triple("r", "rdf:type", "Reviewer")
    assert prefixed.predicate == TYPE_PREDICATE
    assert prefixed.id == type_of("r", "Reviewer").id

    closure = materialize([prefixed], [axiom("a1", "Reviewer", SUBCLASS_OF, "Agent")])
    assert closure.is_entailed("r", "rdf:type", "Agent")


def test_materialize_treats_every_input_fact_as_an_asserted_leaf() -> None:
    closure = materialize([triple("a", "p", "b", derived=True)], [])
    fact = require(closure, "a", "p", "b")
    assert fact.derived is False
    assert closure.explain(fact.id) == ()


# --------------------------------------------------------------------------------------
# The nine derivation rules: each fires ...
# --------------------------------------------------------------------------------------


def test_subclass_transitivity_fires() -> None:
    closure = materialize(
        [],
        [
            axiom("a1", "Reviewer", SUBCLASS_OF, "Agent"),
            axiom("a2", "Agent", SUBCLASS_OF, "Actor"),
        ],
    )
    assert closure.is_entailed("Reviewer", SUBCLASS_OF, "Actor")


def test_type_propagation_fires() -> None:
    closure = materialize(
        [type_of("r", "Reviewer")],
        [axiom("a1", "Reviewer", SUBCLASS_OF, "Agent")],
    )
    assert closure.is_entailed("r", TYPE_PREDICATE, "Agent")


def test_subproperty_transitivity_fires() -> None:
    closure = materialize(
        [],
        [
            axiom("a1", "reviews", SUBPROPERTY_OF, "collaboratesWith"),
            axiom("a2", "collaboratesWith", SUBPROPERTY_OF, "knows"),
        ],
    )
    assert closure.is_entailed("reviews", SUBPROPERTY_OF, "knows")


def test_subproperty_value_propagation_fires() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", SUBPROPERTY_OF, "collaboratesWith")],
    )
    assert closure.is_entailed("r", "collaboratesWith", "doc")


def test_domain_type_fires() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", "domain", "Reviewer")],
    )
    assert closure.is_entailed("r", TYPE_PREDICATE, "Reviewer")


def test_range_type_fires() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", "range", "Artifact")],
    )
    assert closure.is_entailed("doc", TYPE_PREDICATE, "Artifact")


def test_inverse_of_fires() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", "inverseOf", "reviewedBy")],
    )
    assert closure.is_entailed("doc", "reviewedBy", "r")


def test_property_transitivity_fires() -> None:
    closure = materialize(
        [triple("a", "partOf", "b"), triple("b", "partOf", "c")],
        [axiom("a1", "partOf", "transitiveProperty")],
    )
    assert closure.is_entailed("a", "partOf", "c")


def test_property_symmetry_fires() -> None:
    closure = materialize(
        [triple("a", "knows", "b")],
        [axiom("a1", "knows", "symmetricProperty")],
    )
    assert closure.is_entailed("b", "knows", "a")


# --------------------------------------------------------------------------------------
# ... and each stays silent on a near miss
# --------------------------------------------------------------------------------------


def test_subclass_transitivity_does_not_fire_without_a_shared_link() -> None:
    closure = materialize(
        [],
        [
            axiom("a1", "Reviewer", SUBCLASS_OF, "Agent"),
            axiom("a2", "Widget", SUBCLASS_OF, "Actor"),
        ],
    )
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Actor")


def test_type_propagation_does_not_fire_for_an_unrelated_class() -> None:
    closure = materialize(
        [type_of("r", "Reviewer")],
        [axiom("a1", "Widget", SUBCLASS_OF, "Agent")],
    )
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")


def test_subproperty_transitivity_does_not_fire_in_reverse() -> None:
    closure = materialize(
        [],
        [
            axiom("a1", "reviews", SUBPROPERTY_OF, "collaboratesWith"),
            axiom("a2", "collaboratesWith", SUBPROPERTY_OF, "knows"),
        ],
    )
    assert not closure.is_entailed("knows", SUBPROPERTY_OF, "reviews")


def test_subproperty_value_propagation_does_not_run_downhill() -> None:
    closure = materialize(
        [triple("r", "collaboratesWith", "doc")],
        [axiom("a1", "reviews", SUBPROPERTY_OF, "collaboratesWith")],
    )
    assert not closure.is_entailed("r", "reviews", "doc")


def test_domain_type_does_not_fire_for_another_property() -> None:
    closure = materialize(
        [triple("r", "mentions", "doc")],
        [axiom("a1", "reviews", "domain", "Reviewer")],
    )
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Reviewer")


def test_range_type_does_not_type_the_subject() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", "range", "Artifact")],
    )
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Artifact")


def test_inverse_of_does_not_keep_the_argument_order() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [axiom("a1", "reviews", "inverseOf", "reviewedBy")],
    )
    assert not closure.is_entailed("r", "reviewedBy", "doc")


def test_property_transitivity_does_not_fire_without_the_characteristic() -> None:
    closure = materialize(
        [triple("a", "partOf", "b"), triple("b", "partOf", "c")],
        [],
    )
    assert not closure.is_entailed("a", "partOf", "c")


def test_property_symmetry_does_not_fire_without_the_characteristic() -> None:
    closure = materialize([triple("a", "knows", "b")], [])
    assert not closure.is_entailed("b", "knows", "a")


# --------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------


def assert_chain_is_well_formed(closure: Closure, steps: tuple[ProofStep, ...]) -> None:
    """Assert the ordering contract: depths never increase, and the last step rests on leaves."""
    assert steps, "a derived fact must explain itself with at least one step"
    depths = [closure.depth(step.conclusion) for step in steps]
    assert depths == sorted(depths, reverse=True)
    for step in steps:
        assert step.conclusion in closure.facts
        for premise in step.premises:
            assert premise in closure.facts
    last = steps[-1]
    assert all(not closure.facts[premise].derived for premise in last.premises)


def test_three_rule_chain_explains_down_to_asserted_leaves() -> None:
    closure = materialize(
        [triple("r", "reviews", "doc")],
        [
            axiom("a1", "reviews", "domain", "Reviewer"),
            axiom("a2", "Reviewer", SUBCLASS_OF, "Agent"),
            axiom("a3", "Agent", SUBCLASS_OF, "Actor"),
        ],
    )
    goal = require(closure, "r", TYPE_PREDICATE, "Actor")
    assert goal.derived is True

    steps = closure.explain(goal.id)
    assert_chain_is_well_formed(closure, steps)
    assert steps[0].conclusion == goal.id
    rules_used = {step.rule for step in steps}
    assert RULE_DOMAIN_TYPE in rules_used
    assert RULE_TYPE_PROPAGATION in rules_used

    # The route is domain -> Reviewer -> Agent -> Actor, one subsumption step at a time.
    # `subClassOf(Reviewer, Actor)` is still derived by transitivity, but a *derived* schema
    # edge no longer licenses a rule, so it is not what carried this proof. Nothing is lost:
    # type propagation walks the same two asserted edges the transitive one was built from.
    assert closure.is_entailed("Reviewer", SUBCLASS_OF, "Actor")
    assert RULE_SUBCLASS_TRANSITIVITY not in rules_used

    # The chain bottoms out on the asserted property fact and the asserted axiom facts.
    leaf_ids = {
        premise for step in steps for premise in step.premises if not closure.facts[premise].derived
    }
    assert require(closure, "r", "reviews", "doc").id in leaf_ids
    assert require(closure, "reviews", "domain", "Reviewer").id in leaf_ids


def test_asserted_fact_explains_itself_with_an_empty_chain() -> None:
    closure = materialize(
        [type_of("r", "Reviewer")],
        [axiom("a1", "Reviewer", SUBCLASS_OF, "Agent")],
    )
    assert closure.explain(require(closure, "r", TYPE_PREDICATE, "Reviewer").id) == ()


def test_explain_refuses_an_unknown_fact_rather_than_returning_empty() -> None:
    closure = materialize([triple("a", "p", "b")], [])
    with pytest.raises(KeyError):
        closure.explain("0" * 64)


def test_cyclic_derivation_terminates_and_explains_well_foundedly() -> None:
    """`knows` is symmetric, so knows(a,b) and knows(b,a) justify each other.

    Both facts are derived here, so the justification graph genuinely cycles. `explain`
    must still terminate, and must return only the well-founded step -- the one that
    descends towards the asserted leaves rather than back around the loop.
    """
    closure = materialize(
        [triple("a", "friendOf", "b")],
        [
            axiom("a1", "friendOf", SUBPROPERTY_OF, "knows"),
            axiom("a2", "knows", "symmetricProperty"),
        ],
    )
    forward = require(closure, "a", "knows", "b")
    backward = require(closure, "b", "knows", "a")
    assert forward.derived is True
    assert backward.derived is True

    # The cycle is recorded: knows(a,b) has a justification resting on knows(b,a).
    cyclic = [step for step in closure.justifications[forward.id] if backward.id in step.premises]
    assert cyclic, "expected symmetry to record the round-trip justification"

    forward_steps = closure.explain(forward.id)
    assert_chain_is_well_formed(closure, forward_steps)
    assert all(backward.id not in step.premises for step in forward_steps)

    backward_steps = closure.explain(backward.id)
    assert_chain_is_well_formed(closure, backward_steps)
    assert backward_steps[0].rule == RULE_PROPERTY_SYMMETRY


def test_materialize_is_reproducible() -> None:
    facts = [triple("r", "reviews", "doc")]
    axioms = [
        axiom("a1", "reviews", "domain", "Reviewer"),
        axiom("a2", "Reviewer", SUBCLASS_OF, "Agent"),
    ]
    first = materialize(facts, axioms)
    second = materialize(facts, axioms)
    assert set(first.facts) == set(second.facts)
    goal = require(first, "r", TYPE_PREDICATE, "Agent")
    assert first.explain(goal.id) == second.explain(goal.id)


# --------------------------------------------------------------------------------------
# The tier safety rule
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier",
    [OntologyTier.INDUCED_ENFORCING, OntologyTier.INDUCED_CANDIDATE],
)
def test_an_induced_axiom_derives_nothing(tier: OntologyTier) -> None:
    closure = materialize(
        [type_of("r", "Reviewer")],
        [axiom("guess", "Reviewer", SUBCLASS_OF, "Agent", tier=tier)],
    )
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")
    assert [item.source for item in closure.unsupported_axioms] == ["guess"]
    assert tier.value in closure.unsupported_axioms[0].reason


def test_an_induced_schema_fact_is_refused_as_well() -> None:
    """The gate is on the schema statement, not on the container it arrived in."""
    closure = materialize(
        [
            type_of("r", "Reviewer"),
            triple(
                "Reviewer",
                SUBCLASS_OF,
                "Agent",
                tier=OntologyTier.INDUCED_CANDIDATE,
            ),
        ],
        [],
    )
    assert closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    assert len(closure.unsupported_axioms) == 1
    assert "asserted schema" in closure.unsupported_axioms[0].reason


def test_a_conclusion_is_no_stronger_than_its_weakest_premise() -> None:
    closure = materialize(
        [type_of("r", "Reviewer", tier=OntologyTier.INDUCED_CANDIDATE)],
        [axiom("a1", "Reviewer", SUBCLASS_OF, "Agent")],
    )
    derived = require(closure, "r", TYPE_PREDICATE, "Agent")
    assert derived.tier is OntologyTier.INDUCED_CANDIDATE


def test_a_stronger_derivation_upgrades_the_recorded_tier() -> None:
    """The induced derivation is reached first here, so the asserted one must upgrade it."""
    closure = materialize(
        [
            type_of("r", "Auditor", tier=OntologyTier.INDUCED_CANDIDATE),
            type_of("r", "Reviewer"),
        ],
        [
            axiom("a1", "Auditor", SUBCLASS_OF, "Agent"),
            axiom("a2", "Reviewer", SUBCLASS_OF, "Agent"),
        ],
    )
    derived = require(closure, "r", TYPE_PREDICATE, "Agent")
    assert derived.tier is OntologyTier.ASSERTED
    assert len(closure.justifications[derived.id]) == 2


def test_a_repeated_input_fact_is_seeded_once_and_keeps_its_strongest_tier() -> None:
    closure = materialize(
        [
            type_of("r", "Reviewer"),
            type_of("r", "Reviewer", tier=OntologyTier.INDUCED_CANDIDATE),
        ],
        [],
    )
    assert len(closure.facts) == 1
    assert require(closure, "r", TYPE_PREDICATE, "Reviewer").tier is OntologyTier.ASSERTED


# --------------------------------------------------------------------------------------
# Refusals: what the reasoner cannot interpret, it says so about
# --------------------------------------------------------------------------------------


def test_an_unparsable_rule_expression_voids_the_whole_axiom() -> None:
    """The structured fields go inert too: a stripped narrowing clause fires too often.

    `subClassOf(Reviewer, Agent)` here is guarded by a condition the reasoner cannot read.
    Applying the subsumption alone would grant `Agent` unconditionally -- broader than the
    author wrote, which for an authority-granting rule is the shape of incident #22.
    """
    closure = materialize(
        [type_of("r", "Reviewer")],
        [
            axiom(
                "conditional",
                "Reviewer",
                SUBCLASS_OF,
                "Agent",
                rule_expression="only when status == active",
            )
        ],
    )
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")
    assert len(closure.unsupported_axioms) == 1
    assert closure.unsupported_axioms[0].source == "conditional"
    assert "wholly inert" in closure.unsupported_axioms[0].reason


def test_a_parsable_rule_expression_supersedes_the_structured_fields_and_says_so() -> None:
    closure = materialize(
        [triple("a", "p", "b")],
        [
            axiom(
                "rule_and_fields",
                "Reviewer",
                SUBCLASS_OF,
                "Agent",
                rule_expression="p(?x,?y) -> q(?x,?y)",
            )
        ],
    )
    assert closure.is_entailed("a", "q", "b")
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")
    assert len(closure.unsupported_axioms) == 1
    assert "was not applied" in closure.unsupported_axioms[0].reason


@pytest.mark.parametrize(
    ("predicate", "rule_expression", "fragment"),
    [
        # The case that was silently dropped: unrecognised predicate beside a parsing rule.
        ("equivalentClass", "p(?x,?y) -> q(?x,?y)", "not a supported axiom kind in any case"),
        ("subClassOf", "p(?x,?y) -> q(?x,?y)", "was not applied"),
        ("equivalentClass", "", "is not a supported axiom kind"),
    ],
)
def test_an_unapplied_structured_predicate_is_always_reported(
    predicate: str, rule_expression: str, fragment: str
) -> None:
    """Recognised or not, a structured predicate the reasoner did not apply is reported.

    The reporting used to be inverted: only a *recognised* predicate beside a parsing rule
    produced a refusal, so `equivalentClass` -- the spelling the skill-approval scenario
    actually uses -- vanished without trace. An author who believes
    `Approved = Skill n ...` is in force was never contradicted.
    """
    closure = materialize(
        [triple("a", "p", "b")],
        [
            axiom(
                "declared",
                "Reviewer",
                predicate,
                "Agent",
                rule_expression=rule_expression,
            )
        ],
    )
    assert len(closure.unsupported_axioms) == 1
    assert closure.unsupported_axioms[0].source == "declared"
    assert fragment in closure.unsupported_axioms[0].reason
    # The structured triple is inert in every one of these cases.
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")


def test_a_horn_axiom_with_no_structured_predicate_reports_nothing() -> None:
    """The counterpart: an empty `predicate` claims nothing, so there is nothing to refuse."""
    closure = materialize([triple("a", "p", "b")], [horn("clean", "p(?x,?y) -> q(?x,?y)")])
    assert closure.is_entailed("a", "q", "b")
    assert closure.unsupported_axioms == ()


def test_a_validator_style_constraint_axiom_is_refused_not_reinterpreted() -> None:
    """`teach_axiom("R", "Task", "status", "done")` is record validation, not entailment."""
    closure = materialize([type_of("t", "Task")], [axiom("R", "Task", "status", "done")])
    assert len(closure.facts) == 1
    assert "not a supported axiom kind" in closure.unsupported_axioms[0].reason


@pytest.mark.parametrize(
    ("name", "subject", "predicate", "object_value", "fragment"),
    [
        ("missing_subject", "", SUBCLASS_OF, "Agent", "subject_entity"),
        ("missing_object", "Reviewer", SUBCLASS_OF, "", "object_value"),
        ("bad_cardinality", "Review", "maxCardinality", "approvedBy", "property:N"),
        ("noisy_unary", "knows", "symmetricProperty", "sometimes", "takes no argument"),
    ],
)
def test_malformed_axiom_arguments_are_refused(
    name: str, subject: str, predicate: str, object_value: str, fragment: str
) -> None:
    closure = materialize([], [axiom(name, subject, predicate, object_value)])
    assert closure.facts == {}
    assert len(closure.unsupported_axioms) == 1
    assert closure.unsupported_axioms[0].source == name
    assert fragment in closure.unsupported_axioms[0].reason


def test_a_malformed_cardinality_fact_is_refused() -> None:
    closure = materialize([triple("Review", "maxCardinality", "approvedBy")], [])
    assert len(closure.unsupported_axioms) == 1
    assert "property:N" in closure.unsupported_axioms[0].reason


def test_axiom_kind_spellings_are_normalised() -> None:
    assert normalize_axiom_kind("rdfs:subClassOf") == SUBCLASS_OF
    assert normalize_axiom_kind("sub_class_of") == SUBCLASS_OF
    assert normalize_axiom_kind("OWL: transitive") == "transitiveProperty"
    assert normalize_axiom_kind("inverse-functional") == "inverseFunctionalProperty"
    assert normalize_axiom_kind("subclass") is None


def test_a_normalised_axiom_kind_still_derives() -> None:
    closure = materialize(
        [type_of("r", "Reviewer")],
        [axiom("a1", "Reviewer", "rdfs:sub_class_of", "Agent")],
    )
    assert closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    assert closure.unsupported_axioms == ()


# --------------------------------------------------------------------------------------
# The thirteenth capability: safe Horn rules
# --------------------------------------------------------------------------------------

APPROVED_RULE = (
    "Skill(?s) ^ hasVerdict(?s,?v) ^ rdf:type(?v,'AuditVerdict.APPROVE') "
    "^ verdictSubject(?v,?h) ^ contentSha256(?s,?h) -> rdf:type(?s,'Approved')"
)
AUDITED_HASH = "sha256:v1"
EDITED_HASH = "sha256:v2"


def approval_facts(
    content_hash: str,
    verdict_hash: str,
    verdict: str = "AuditVerdict.APPROVE",
) -> list[Fact]:
    """Build the incident-#22 fixture: a skill, its content hash, and an auditor verdict."""
    return [
        type_of("skill:evil-echo", "Skill"),
        triple("skill:evil-echo", "contentSha256", content_hash),
        triple("skill:evil-echo", "hasVerdict", "verdict:v1"),
        type_of("verdict:v1", verdict),
        triple("verdict:v1", "verdictSubject", verdict_hash),
    ]


def horn(name: str, expression: str, tier: OntologyTier = OntologyTier.ASSERTED) -> OntologyAxiom:
    """Build a Horn-rule axiom, whose semantics live entirely in `rule_expression`."""
    return axiom(name, subject_entity=name, predicate="", rule_expression=expression, tier=tier)


def test_a_horn_rule_derives_a_class_from_a_conjunctive_body() -> None:
    closure = materialize(
        approval_facts(AUDITED_HASH, AUDITED_HASH), [horn("approved", APPROVED_RULE)]
    )
    assert closure.is_entailed("skill:evil-echo", TYPE_PREDICATE, "Approved")
    assert closure.unsupported_axioms == ()


def test_the_join_binds_the_verdict_to_this_content() -> None:
    """The `?h` join is the load-bearing part: an audit cannot be inherited by other code."""
    closure = materialize(
        approval_facts(EDITED_HASH, AUDITED_HASH),
        [horn("approved", APPROVED_RULE)],
    )
    assert not closure.is_entailed("skill:evil-echo", TYPE_PREDICATE, "Approved")


def test_a_reject_verdict_does_not_entail_approval() -> None:
    closure = materialize(
        approval_facts(AUDITED_HASH, AUDITED_HASH, verdict="AuditVerdict.REJECT"),
        [horn("approved", APPROVED_RULE)],
    )
    assert not closure.is_entailed("skill:evil-echo", TYPE_PREDICATE, "Approved")


def test_a_self_asserted_status_cannot_buy_approval() -> None:
    """Incident #22 in ontology terms: the artifact declares its own status and it changes nothing."""
    attack = triple("skill:evil-echo", "status", "active")
    rejected = materialize(
        [*approval_facts(AUDITED_HASH, AUDITED_HASH, verdict="AuditVerdict.REJECT"), attack],
        [horn("approved", APPROVED_RULE)],
    )
    assert not rejected.is_entailed("skill:evil-echo", TYPE_PREDICATE, "Approved")
    # The fact is legitimately in the closure; it simply has no path to Approved.
    assert rejected.is_entailed("skill:evil-echo", "status", "active")


def test_a_horn_firing_explains_itself_with_the_verdict_and_the_rule() -> None:
    closure = materialize(
        approval_facts(AUDITED_HASH, AUDITED_HASH), [horn("approved", APPROVED_RULE)]
    )
    goal = require(closure, "skill:evil-echo", TYPE_PREDICATE, "Approved")
    steps = closure.explain(goal.id)
    assert_chain_is_well_formed(closure, steps)
    assert len(steps) == 1
    assert steps[0].rule == f"{RULE_HORN}:approved"

    cited = {closure.facts[premise].triple for premise in steps[0].premises}
    assert ("verdict:v1", TYPE_PREDICATE, "AuditVerdict.APPROVE") in cited
    assert ("verdict:v1", "verdictSubject", AUDITED_HASH) in cited
    assert ("skill:evil-echo", "contentSha256", AUDITED_HASH) in cited
    # The rule itself is cited, so the explanation names what read the evidence.
    assert any(triple_[1] == HORN_RULE for triple_ in cited)


def test_an_induced_horn_rule_derives_nothing() -> None:
    closure = materialize(
        approval_facts(AUDITED_HASH, AUDITED_HASH),
        [horn("guess", APPROVED_RULE, tier=OntologyTier.INDUCED_ENFORCING)],
    )
    assert not closure.is_entailed("skill:evil-echo", TYPE_PREDICATE, "Approved")
    assert closure.unsupported_axioms[0].source == "guess"


def test_an_induced_horn_rule_fact_is_refused_by_the_index() -> None:
    parsed = parse_horn_rule("p(?x,?y) -> q(?x,?y)")
    assert parsed.rule is not None
    closure = materialize(
        [
            triple("a", "p", "b"),
            triple(
                "smuggled",
                HORN_RULE,
                canonical_horn_text(parsed.rule),
                tier=OntologyTier.INDUCED_CANDIDATE,
            ),
        ],
        [],
    )
    assert not closure.is_entailed("a", "q", "b")
    assert "asserted schema" in closure.unsupported_axioms[0].reason


def test_a_horn_rule_conclusion_is_no_stronger_than_its_weakest_premise() -> None:
    closure = materialize(
        [triple("a", "p", "b", tier=OntologyTier.INDUCED_CANDIDATE)],
        [horn("r", "p(?x,?y) -> q(?x,?y)")],
    )
    assert require(closure, "a", "q", "b").tier is OntologyTier.INDUCED_CANDIDATE


def test_a_recursive_horn_rule_reaches_a_fixed_point() -> None:
    """Ancestor closure: the classic recursive rule, which must terminate and stay explainable."""
    closure = materialize(
        [
            triple("a", "parentOf", "b"),
            triple("b", "parentOf", "c"),
            triple("c", "parentOf", "d"),
        ],
        [
            horn("base", "parentOf(?x,?y) -> ancestorOf(?x,?y)"),
            horn("step", "ancestorOf(?x,?y) ^ parentOf(?y,?z) -> ancestorOf(?x,?z)"),
        ],
    )
    assert closure.is_entailed("a", "ancestorOf", "d")
    assert not closure.is_entailed("d", "ancestorOf", "a")
    goal = require(closure, "a", "ancestorOf", "d")
    assert_chain_is_well_formed(closure, closure.explain(goal.id))


# --------------------------------------------------------------------------------------
# Route D6: schema lifting
#
# A rule that derives schema hands every writer of an ordinary data triple the power to
# mint schema, and the fixed rules then propagate along it. Two independent guards close
# the route: no Horn head may name a schema predicate, and no *derived* schema edge may
# license a rule. The second is not redundant -- the last test here reaches D6 with no Horn
# rule at all, where the head guard has nothing to look at.
# --------------------------------------------------------------------------------------


def test_a_horn_rule_may_not_derive_schema_that_the_fixed_rules_would_then_use() -> None:
    """Formerly asserted as a feature: deriving `subClassOf` to feed type propagation.

    It is route D6. `promotedTo` is an ordinary predicate with no authority of its own, so
    a rule lifting it into `subClassOf` lets anyone who can write one triple grant a class
    membership -- which is the whole tier model, undone by a rule that reads as a taxonomy
    convenience.
    """
    closure = materialize(
        [type_of("r", "Reviewer"), triple("Reviewer", "promotedTo", "Agent")],
        [horn("promote", "promotedTo(?a,?b) -> subClassOf(?a,?b)")],
    )
    assert not closure.is_entailed("Reviewer", SUBCLASS_OF, "Agent")
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    assert len(closure.unsupported_axioms) == 1
    assert closure.unsupported_axioms[0].source == "promote"
    assert "may not derive schema" in closure.unsupported_axioms[0].reason
    assert SUBCLASS_OF in closure.unsupported_axioms[0].reason


def test_schema_lifting_cannot_reach_approved_without_a_verdict() -> None:
    """The D6 reproducer: one benign taxonomy rule plus one attacker-controlled triple.

    `declaresCategory` is a plain predicate on which the attacker may write freely. Lifted
    into `subClassOf`, it would make `Skill` a subclass of `Approved` and type propagation
    would then admit every skill -- with no audit verdict in the closure at all.
    """
    lift = horn("lift", "declaresCategory(?c,?parent) -> subClassOf(?c,?parent)")
    closure = materialize(
        [
            type_of("skill:evil", "Skill"),
            triple("Skill", "declaresCategory", "Approved"),
        ],
        [lift],
    )
    # There is no verdict anywhere in this closure; approval must be unreachable.
    assert not any(fact.predicate == "hasVerdict" for fact in closure.facts.values())
    assert not closure.is_entailed("skill:evil", TYPE_PREDICATE, "Approved")
    assert not closure.is_entailed("Skill", SUBCLASS_OF, "Approved")
    assert [item.source for item in closure.unsupported_axioms] == ["lift"]
    assert "may not derive schema" in closure.unsupported_axioms[0].reason


@pytest.mark.parametrize("predicate", sorted(SCHEMA_PREDICATES))
def test_every_schema_predicate_is_refused_as_a_horn_head(predicate: str) -> None:
    """Enumerated from the frozenset, so a predicate added later is covered on arrival."""
    parsed = parse_horn_rule(f"p(?x,?y) -> {predicate}(?x,?y)")
    assert parsed.rule is None, f"'{predicate}' was accepted as a head predicate"
    assert "may not derive schema" in parsed.reason
    assert predicate in parsed.reason


def test_a_derived_schema_fact_does_not_license_further_derivation() -> None:
    """D6 without a Horn rule: `subPropertyOf` into a schema predicate lifts the same way.

    Every axiom here is asserted and every fixed rule is the stock one, so the tier gate
    sees nothing wrong: the derived `subClassOf` inherits `asserted` from asserted
    premises. Only the `derived` check refuses it, which is why that check is not merely
    defence in depth -- it is the sole guard on this route.
    """
    closure = materialize(
        [type_of("r", "Reviewer"), triple("Reviewer", "promotedTo", "Agent")],
        [axiom("lift", "promotedTo", SUBPROPERTY_OF, SUBCLASS_OF)],
    )
    # The subsumption edge is derived -- that much is ordinary subproperty propagation.
    lifted = require(closure, "Reviewer", SUBCLASS_OF, "Agent")
    assert lifted.derived is True
    assert lifted.tier is OntologyTier.ASSERTED, "tier alone cannot catch this"

    # But it licenses nothing, so the membership it was manufactured to grant never lands.
    assert not closure.is_entailed("r", TYPE_PREDICATE, "Agent")
    reasons = [item.reason for item in closure.unsupported_axioms]
    assert any("was derived, not asserted" in reason for reason in reasons)


def test_a_transitively_derived_subclass_edge_is_refused_but_costs_no_entailment() -> None:
    """The gate's one real cost, pinned so its scope stays visible.

    `subClassOf(Reviewer, Actor)` is derived, so it no longer licenses type propagation and
    is reported as refused. No entailment is lost by that: propagation walks the two
    asserted edges one step at a time and reaches `Actor` regardless.
    """
    closure = materialize(
        [type_of("r", "Reviewer")],
        [
            axiom("a1", "Reviewer", SUBCLASS_OF, "Agent"),
            axiom("a2", "Agent", SUBCLASS_OF, "Actor"),
        ],
    )
    assert closure.is_entailed("Reviewer", SUBCLASS_OF, "Actor")
    assert closure.is_entailed("r", TYPE_PREDICATE, "Actor")
    assert require(closure, "Reviewer", SUBCLASS_OF, "Actor").derived is True
    assert any("was derived, not asserted" in item.reason for item in closure.unsupported_axioms)


def test_a_variable_repeated_inside_one_atom_must_agree() -> None:
    closure = materialize(
        [triple("a", "p", "a"), triple("b", "p", "c")],
        [horn("loop", "p(?x,?x) -> selfLoop(?x,?x)")],
    )
    assert closure.is_entailed("a", "selfLoop", "a")
    assert not closure.is_entailed("b", "selfLoop", "b")


def test_a_ground_head_needs_no_variables() -> None:
    """A head with no variables is trivially safe: nothing needs binding."""
    closure = materialize(
        [triple("a", "p", "b")],
        [horn("flag", "p(?x,?y) -> systemState(alerted,true)")],
    )
    assert closure.is_entailed("alerted", "systemState", "true")


def test_a_quoted_constant_may_contain_spaces() -> None:
    closure = materialize(
        [triple("a", "label", "needs review")],
        [horn("flagged", "label(?x,'needs review') -> type(?x,'Flagged')")],
    )
    assert closure.is_entailed("a", TYPE_PREDICATE, "Flagged")


def test_both_arrow_directions_express_the_same_rule() -> None:
    body_first = parse_horn_rule("p(?x,?y) -> q(?x,?y)")
    head_first = parse_horn_rule("q(?x,?y) :- p(?x,?y)")
    fat_arrow = parse_horn_rule("p(?x,?y) => q(?x,?y)")
    left_arrow = parse_horn_rule("q(?x,?y) <- p(?x,?y)")
    assert body_first.rule is not None
    assert head_first.rule == body_first.rule
    assert fat_arrow.rule == body_first.rule
    assert left_arrow.rule == body_first.rule


def test_equivalent_spellings_of_a_rule_collapse_to_one_fact() -> None:
    """Canonical rule text is what keeps a rule one fact rather than several."""
    spellings = [
        "Skill(?s) & hasVerdict(?s,?v) -> rdf:type(?s,'Seen')",
        "rdf:type(?s,Skill) , hasVerdict(?s, ?v)  ->  type(?s, 'Seen')",
        "type(?s,'Seen') :- Skill(?s) ^ hasVerdict(?s,?v)",
    ]
    parsed = [parse_horn_rule(text) for text in spellings]
    assert all(item.rule is not None for item in parsed)
    canonical = {canonical_horn_text(item.rule) for item in parsed if item.rule is not None}
    assert len(canonical) == 1


def test_a_class_atom_means_a_type_atom() -> None:
    parsed = parse_horn_rule("Skill(?s) -> type(?s,'Seen')")
    assert parsed.rule is not None
    assert parsed.rule.body[0].predicate == TYPE_PREDICATE
    assert parsed.rule.body[0].object == HornTerm(value="Skill", is_variable=False)
    assert parsed.rule.variables() == frozenset({"s"})


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("", "empty"),
        ("p(?x,?y)", "no implication arrow"),
        ("p(?x,?y) -> q(?z,?y)", "unsafe"),
        ("-> q(?x,?y)", "empty conjunct"),
        ("p(?x,?y) -> ", "head atom '' does not parse"),
        ("p(?x,?y) ^ -> q(?x,?y)", "empty conjunct"),
        ("p(?x,?y) -> q(?x,?y) ^ r(?x,?y)", "exactly one atom"),
        ("p(?x,?y) -> q(?x,?y) -> r(?x,?y)", "more than one implication arrow"),
        ("not p(?x,?y) -> q(?x,?y)", "negation, disjunction and comparison"),
        ("p(?x,?y) ; r(?x,?y) -> q(?x,?y)", "negation, disjunction and comparison"),
        ("p(?x,?y) ^ ?x = ?y -> q(?x,?y)", "negation, disjunction and comparison"),
        ("p(?x,?y) -> hornRule(?x,?y)", "may not derive schema"),
        ("p(?x,?y -> q(?x,?y)", "unbalanced parentheses or quotes"),
        ("p(?x,?y)) -> q(?x,?y)", "unbalanced parentheses or quotes"),
        ("p(?x,?y,?z) -> q(?x,?y)", "does not parse"),
        ("p() -> q(?x,?y)", "does not parse"),
        ("p(?) -> q(?x,?y)", "does not parse"),
        # An unterminated quote swallows the arrow, so the refusal names the missing arrow.
        ("p('un'quoted',?y) -> q(?x,?y)", "no implication arrow"),
        ("p(bare word,?y) -> q(?x,?y)", "does not parse"),
        ("9bad(?x,?y) -> q(?x,?y)", "does not parse"),
    ],
)
def test_the_grammar_refuses_what_it_does_not_cover(expression: str, fragment: str) -> None:
    parsed = parse_horn_rule(expression)
    assert parsed.rule is None
    assert fragment in parsed.reason


def test_a_refused_rule_expression_is_reported_on_the_closure() -> None:
    closure = materialize([triple("a", "p", "b")], [horn("unsafe", "p(?x,?y) -> q(?z,?y)")])
    assert closure.facts == {require(closure, "a", "p", "b").id: require(closure, "a", "p", "b")}
    assert len(closure.unsupported_axioms) == 1
    assert "unsafe" in closure.unsupported_axioms[0].reason


def test_a_malformed_horn_rule_fact_is_refused_by_the_index() -> None:
    closure = materialize([triple("broken", HORN_RULE, "this is not a rule")], [])
    assert len(closure.unsupported_axioms) == 1
    assert "does not parse as a Horn rule" in closure.unsupported_axioms[0].reason


# --------------------------------------------------------------------------------------
# Consistency detection
# --------------------------------------------------------------------------------------


def test_disjointness_violation_names_both_type_facts() -> None:
    closure = materialize(
        [type_of("t", "Completed"), type_of("t", "Cancelled")],
        [axiom("a1", "Completed", "disjointWith", "Cancelled")],
    )
    findings = check_consistency(closure)
    assert len(findings) == 1
    assert findings[0].kind == KIND_DISJOINT
    assert set(findings[0].facts) == {
        require(closure, "t", TYPE_PREDICATE, "Completed").id,
        require(closure, "t", TYPE_PREDICATE, "Cancelled").id,
        require(closure, "Completed", "disjointWith", "Cancelled").id,
    }
    assert "disjoint" in findings[0].explanation


def test_disjointness_holds_when_only_one_membership_is_entailed() -> None:
    closure = materialize(
        [type_of("t", "Completed")],
        [axiom("a1", "Completed", "disjointWith", "Cancelled")],
    )
    assert check_consistency(closure) == ()


def test_disjointness_is_detected_through_a_derived_type() -> None:
    closure = materialize(
        [type_of("t", "Cancelled"), triple("t", "finishedBy", "u")],
        [
            axiom("a1", "finishedBy", "domain", "Completed"),
            axiom("a2", "Completed", "disjointWith", "Cancelled"),
        ],
    )
    findings = check_consistency(closure)
    assert [item.kind for item in findings] == [KIND_DISJOINT]


def test_maximum_cardinality_violation_is_reported() -> None:
    closure = materialize(
        [
            type_of("rev", "Review"),
            triple("rev", "approvedBy", "alice"),
            triple("rev", "approvedBy", "bob"),
        ],
        [axiom("a1", "Review", "maxCardinality", "approvedBy:1")],
    )
    findings = check_consistency(closure)
    assert len(findings) == 1
    assert findings[0].kind == KIND_CARDINALITY
    assert require(closure, "rev", "approvedBy", "alice").id in findings[0].facts
    assert require(closure, "rev", "approvedBy", "bob").id in findings[0].facts
    assert findings[0].facts[-1] == require(closure, "Review", "maxCardinality", "approvedBy:1").id


def test_maximum_cardinality_is_satisfied_at_the_limit() -> None:
    closure = materialize(
        [type_of("rev", "Review"), triple("rev", "approvedBy", "alice")],
        [axiom("a1", "Review", "maxCardinality", "approvedBy:1")],
    )
    assert check_consistency(closure) == ()


def test_minimum_cardinality_violation_is_reported() -> None:
    closure = materialize(
        [type_of("rev", "Review"), triple("rev", "approvedBy", "alice")],
        [axiom("a1", "Review", "minCardinality", "approvedBy:2")],
    )
    findings = check_consistency(closure)
    assert len(findings) == 1
    assert findings[0].kind == KIND_CARDINALITY
    assert "at least 2" in findings[0].explanation


def test_minimum_cardinality_is_satisfied_at_the_floor() -> None:
    closure = materialize(
        [
            type_of("rev", "Review"),
            triple("rev", "approvedBy", "alice"),
            triple("rev", "approvedBy", "bob"),
        ],
        [axiom("a1", "Review", "minCardinality", "approvedBy:2")],
    )
    assert check_consistency(closure) == ()


def test_minimum_cardinality_ignores_individuals_outside_the_class() -> None:
    closure = materialize(
        [type_of("other", "Draft")],
        [axiom("a1", "Review", "minCardinality", "approvedBy:2")],
    )
    assert check_consistency(closure) == ()


def test_functional_violation_names_both_values() -> None:
    closure = materialize(
        [triple("t", "hasOwner", "alice"), triple("t", "hasOwner", "bob")],
        [axiom("a1", "hasOwner", "functionalProperty")],
    )
    findings = check_consistency(closure)
    assert len(findings) == 1
    assert findings[0].kind == KIND_FUNCTIONAL
    assert set(findings[0].facts[:2]) == {
        require(closure, "t", "hasOwner", "alice").id,
        require(closure, "t", "hasOwner", "bob").id,
    }


def test_functional_property_with_one_value_is_consistent() -> None:
    closure = materialize(
        [triple("t", "hasOwner", "alice"), triple("u", "hasOwner", "bob")],
        [axiom("a1", "hasOwner", "functionalProperty")],
    )
    assert check_consistency(closure) == ()


def test_inverse_functional_violation_names_both_subjects() -> None:
    closure = materialize(
        [triple("p1", "hasBadge", "B-9"), triple("p2", "hasBadge", "B-9")],
        [axiom("a1", "hasBadge", "inverseFunctionalProperty")],
    )
    findings = check_consistency(closure)
    assert len(findings) == 1
    assert findings[0].kind == KIND_INVERSE_FUNCTIONAL
    assert set(findings[0].facts[:2]) == {
        require(closure, "p1", "hasBadge", "B-9").id,
        require(closure, "p2", "hasBadge", "B-9").id,
    }


def test_inverse_functional_property_with_distinct_objects_is_consistent() -> None:
    closure = materialize(
        [triple("p1", "hasBadge", "B-9"), triple("p2", "hasBadge", "B-10")],
        [axiom("a1", "hasBadge", "inverseFunctionalProperty")],
    )
    assert check_consistency(closure) == ()


def test_entailment_and_consistency_are_separate_questions() -> None:
    """The documented contract, pinned: `is_entailed` does not account for a contradiction.

    A skill is typed both `Approved` and `Rejected`, which are declared disjoint. The
    closure knows this is contradictory, and `is_entailed` still answers `True` for
    `Approved` -- because membership is a lookup, and which of two conflicting facts to
    retract is a policy question the reasoner cannot answer on the caller's behalf.

    This test exists to make the obligation break loudly if the contract is ever changed
    silently in either direction. A caller granting authority must ask both questions.
    """
    closure = materialize(
        [type_of("skill:x", "Approved"), type_of("skill:x", "Rejected")],
        [axiom("a1", "Approved", "disjointWith", "Rejected")],
    )
    assert closure.is_entailed("skill:x", TYPE_PREDICATE, "Approved") is True

    findings = check_consistency(closure)
    assert [item.kind for item in findings] == [KIND_DISJOINT]
    # Asking only the first question would admit a skill the closure knows is contradictory.


def test_an_induced_disjointness_axiom_detects_nothing() -> None:
    closure = materialize(
        [type_of("t", "Completed"), type_of("t", "Cancelled")],
        [
            axiom(
                "guess",
                "Completed",
                "disjointWith",
                "Cancelled",
                tier=OntologyTier.INDUCED_CANDIDATE,
            )
        ],
    )
    assert check_consistency(closure) == ()


# --------------------------------------------------------------------------------------
# Closure invariants
# --------------------------------------------------------------------------------------


def test_a_derived_fact_without_a_justification_is_rejected() -> None:
    orphan = triple("a", "p", "b", derived=True)
    with pytest.raises(OntologyViolationError, match="no justification"):
        Closure(facts={orphan.id: orphan})


def test_a_justification_citing_an_absent_premise_is_rejected() -> None:
    conclusion = triple("a", "p", "b", derived=True)
    step = ProofStep(rule="invented", premises=("0" * 64,), conclusion=conclusion.id)
    with pytest.raises(OntologyViolationError, match="not in the closure"):
        Closure(facts={conclusion.id: conclusion}, justifications={conclusion.id: (step,)})


def test_a_justification_filed_under_the_wrong_fact_is_rejected() -> None:
    leaf = triple("a", "p", "b")
    conclusion = triple("b", "p", "a", derived=True)
    step = ProofStep(rule="invented", premises=(leaf.id,), conclusion=leaf.id)
    with pytest.raises(OntologyViolationError, match="concludes"):
        Closure(
            facts={leaf.id: leaf, conclusion.id: conclusion},
            justifications={conclusion.id: (step,)},
        )


def test_a_justification_for_an_unknown_fact_is_rejected() -> None:
    leaf = triple("a", "p", "b")
    step = ProofStep(rule="invented", premises=(leaf.id,), conclusion="0" * 64)
    with pytest.raises(OntologyViolationError, match="unknown fact"):
        Closure(facts={leaf.id: leaf}, justifications={"0" * 64: (step,)})


def test_a_fact_round_trips_through_validation_unchanged() -> None:
    original = triple("a", "p", "b")
    assert Fact.model_validate(original) == original
    with pytest.raises(ValidationError):
        Fact.model_validate({"subject": "a", "object": "b"})
    with pytest.raises(ValidationError):
        Fact.model_validate(["a", "p", "b"])


def test_a_purely_cyclic_support_explains_nothing_rather_than_looping() -> None:
    """A hand-built closure whose only support is a cycle: `explain` must still terminate.

    `materialize` cannot produce this -- forward chaining always records a well-founded
    step -- but `Closure` accepts justifications from anywhere, so the guarantee is tested
    on the worst input rather than assumed. `b` and `c` justify each other and nothing
    else, so neither is grounded; `a` is grounded through the leaf and must explain itself
    without following its second, cyclic justification.
    """
    leaf = triple("leaf", "p", "x")
    grounded = triple("a", "p", "x", derived=True)
    left = triple("b", "p", "x", derived=True)
    right = triple("c", "p", "x", derived=True)
    closure = Closure(
        facts={
            leaf.id: leaf,
            grounded.id: grounded,
            left.id: left,
            right.id: right,
        },
        justifications={
            grounded.id: (
                ProofStep(rule="from_leaf", premises=(leaf.id,), conclusion=grounded.id),
                ProofStep(rule="from_cycle", premises=(left.id,), conclusion=grounded.id),
            ),
            left.id: (ProofStep(rule="cycle", premises=(right.id,), conclusion=left.id),),
            right.id: (ProofStep(rule="cycle", premises=(left.id,), conclusion=right.id),),
        },
    )
    grounded_chain = closure.explain(grounded.id)
    assert [step.rule for step in grounded_chain] == ["from_leaf"]
    assert closure.explain(left.id) == ()
    assert closure.explain(right.id) == ()
    assert closure.depth(grounded.id) == 1
    assert closure.depth(left.id) == 0


def test_a_diamond_derivation_reports_each_step_once() -> None:
    """Two routes to the same conclusion share a premise; the shared step is not duplicated."""
    closure = materialize(
        [type_of("r", "Reviewer")],
        [
            axiom("a1", "Reviewer", SUBCLASS_OF, "Agent"),
            axiom("a2", "Reviewer", SUBCLASS_OF, "Auditor"),
            axiom("a3", "Agent", SUBCLASS_OF, "Actor"),
            axiom("a4", "Auditor", SUBCLASS_OF, "Actor"),
        ],
    )
    goal = require(closure, "r", TYPE_PREDICATE, "Actor")
    steps = closure.explain(goal.id)
    assert_chain_is_well_formed(closure, steps)
    assert len(steps) == len(set(steps))


def test_an_empty_closure_is_consistent_and_entails_nothing() -> None:
    closure = materialize([], [])
    assert closure.facts == {}
    assert closure.justifications == {}
    assert closure.unsupported_axioms == ()
    assert check_consistency(closure) == ()
    assert closure.is_entailed("a", "p", "b") is False
    assert closure.get("a", "p", "b") is None
    assert closure.depth("0" * 64) == 0
