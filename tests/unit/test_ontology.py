"""Unit tests for tiered Ontology Engine, content-hash validation, and contradiction handling."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from uclone_x.errors import (
    OntologyContradictionError,
    OntologyPromotionError,
    OntologyRetractionBlockedError,
    OntologyViolationError,
    UnparseableDirectiveError,
)
from uclone_x.ontology.engine import (
    OntologyEngine,
    OntologyInducer,
    OntologyService,
)
from uclone_x.ontology.models import (
    EntitySchema,
    EvidenceRecord,
    OntologyAxiom,
    OntologyConcept,
    OntologyRelation,
    OntologyTier,
    OntologyValidationResult,
    RelationSchema,
    ValidationResult,
    compute_axiom_hash,
    compute_concept_hash,
    compute_relation_hash,
    normalize_tier,
    tier_to_precedence,
)


def test_ontology_tier_and_precedence() -> None:
    assert OntologyTier.ASSERTED == "asserted"
    assert OntologyTier.INDUCED_ENFORCING == "induced_enforcing"
    assert OntologyTier.INDUCED_CANDIDATE == "induced_candidate"

    # Hyphenated aliases via _missing_ and normalize_tier
    assert OntologyTier("induced-candidate") == OntologyTier.INDUCED_CANDIDATE
    assert OntologyTier("induced-enforcing") == OntologyTier.INDUCED_ENFORCING
    assert OntologyTier("INDUCED_CANDIDATE") == OntologyTier.INDUCED_CANDIDATE
    assert OntologyTier("INDUCED-CANDIDATE") == OntologyTier.INDUCED_CANDIDATE
    assert normalize_tier("induced-candidate") == OntologyTier.INDUCED_CANDIDATE
    assert normalize_tier("induced-enforcing") == OntologyTier.INDUCED_ENFORCING
    assert normalize_tier(OntologyTier.ASSERTED) == OntologyTier.ASSERTED

    assert tier_to_precedence(OntologyTier.ASSERTED) == 100
    assert tier_to_precedence(OntologyTier.INDUCED_ENFORCING) == 50
    assert tier_to_precedence(OntologyTier.INDUCED_CANDIDATE) == 10
    assert tier_to_precedence("asserted") == 100
    assert tier_to_precedence("induced-enforcing") == 50
    assert tier_to_precedence("induced-candidate") == 10

    # P6 fail-fast: unrecognised tier strings or invalid types MUST raise ValueError, NEVER fallback
    with pytest.raises(ValueError, match="Unrecognised ontology tier"):
        tier_to_precedence("unknown")
    with pytest.raises(ValueError, match="Unrecognised ontology tier"):
        tier_to_precedence("arbitrary_tier")
    with pytest.raises(ValueError, match="Unrecognised ontology tier"):
        tier_to_precedence(12345)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        OntologyTier("arbitrary_tier")
    with pytest.raises(ValueError):
        OntologyTier("unknown")
    with pytest.raises(ValueError, match="Unrecognised ontology tier"):
        normalize_tier("arbitrary_tier")
    with pytest.raises(ValueError, match="Unrecognised ontology tier"):
        normalize_tier(999)  # type: ignore[arg-type]


def test_evidence_record_immutability_and_defaults() -> None:
    ev = EvidenceRecord(
        observation_count=3,
        source_session="session_123",
        model_id="gemini-2.5-pro",
        confidence=0.95,
    )
    assert ev.observation_count == 3
    assert ev.source_session == "session_123"
    assert ev.model_id == "gemini-2.5-pro"
    assert ev.confidence == 0.95
    assert ev.contradicting_observations == ()
    assert ev.first_seen is not None
    assert ev.last_seen is not None

    with pytest.raises(ValidationError):
        ev.observation_count = 5  # type: ignore[misc]

    with pytest.raises(ValidationError):
        EvidenceRecord(confidence=1.5)  # type: ignore[call-arg]


def test_ontology_models_and_hash_computation() -> None:
    c_hash = compute_concept_hash(
        name="User",
        parent_type="Entity",
        attributes={"email": "str", "age": "int"},
        required_fields=["email"],
    )
    assert isinstance(c_hash, str) and len(c_hash) == 64

    # Order of attributes or required_fields does not change hash
    c_hash_reordered = compute_concept_hash(
        name="User",
        parent_type="Entity",
        attributes={"age": "int", "email": "str"},
        required_fields=["email"],
    )
    assert c_hash == c_hash_reordered

    concept = OntologyConcept(
        name="User",
        parent_type="Entity",
        attributes={"email": "str", "age": "int"},
        required_fields=("email",),
    )
    assert concept.content_hash == c_hash
    assert concept.precedence == 100
    assert concept.tier == OntologyTier.ASSERTED

    rel_hash = compute_relation_hash(
        source_entity="User",
        predicate="owns",
        target_entity="Account",
        is_directed=True,
    )
    relation = OntologyRelation(
        source_entity="User",
        predicate="owns",
        target_entity="Account",
    )
    assert relation.content_hash == rel_hash
    assert relation.precedence == 100

    ax_hash = compute_axiom_hash(
        name="RequireActive",
        subject_entity="Account",
        predicate="status",
        object_value="active",
        rule_expression="",
    )
    axiom = OntologyAxiom(
        name="RequireActive",
        subject_entity="Account",
        predicate="status",
        object_value="active",
    )
    assert axiom.content_hash == ax_hash
    assert axiom.precedence == 100

    # ValidationResult
    val = ValidationResult(is_valid=True)
    assert val.is_valid is True
    assert val.errors == ()
    assert val.warnings == ()


def test_backward_compatibility_aliases() -> None:
    entity = EntitySchema(name="Repo", description="Git repository")
    assert isinstance(entity, OntologyConcept)
    assert entity.name == "Repo"

    rel = RelationSchema(source_entity="Repo", predicate="has", target_entity="Branch")
    assert isinstance(rel, OntologyRelation)
    assert rel.predicate == "has"

    val = OntologyValidationResult(is_valid=True)
    assert isinstance(val, ValidationResult)
    assert val.is_valid is True


def test_ontology_engine_teaching_and_content_hash() -> None:
    engine = OntologyEngine(agent_id="agt_sec")
    initial_hash = engine.content_hash

    # Teach concept
    c = engine.teach_concept(
        name="SecurityPolicy",
        description="Auth policy",
        attributes={"algorithm": "str", "key_size": "int"},
        required_fields=["algorithm"],
    )
    assert c.name == "SecurityPolicy"
    assert c.tier == OntologyTier.ASSERTED
    assert c.precedence == 100
    assert engine.version == 2
    hash_after_c = engine.content_hash
    assert hash_after_c != initial_hash

    # Teach relation
    r = engine.teach_relation(
        source_entity="SecurityPolicy",
        predicate="enforces",
        target_entity="JWTPayload",
    )
    assert r.predicate == "enforces"
    assert engine.version == 3
    hash_after_r = engine.content_hash
    assert hash_after_r != hash_after_c

    # Teach axiom
    ax = engine.teach_axiom(
        name="RequireHS256",
        subject_entity="SecurityPolicy",
        predicate="algorithm",
        object_value="HS256",
    )
    assert ax.name == "RequireHS256"
    assert engine.version == 4
    hash_after_ax = engine.content_hash
    assert hash_after_ax != hash_after_r

    # Inducing a candidate must NOT alter the asserted content_hash!
    engine.induce_concept(
        name="CandidatePolicy",
        description="Candidate only",
        attributes={"temp": "str"},
    )
    assert engine.content_hash == hash_after_ax


def test_ontology_engine_directives() -> None:
    engine = OntologyEngine()

    # Directive format 1: Concept
    c1 = engine.teach_directive(
        "Concept User extends BaseEntity with attributes name:str, age:int requires name"
    )
    assert isinstance(c1, OntologyConcept)
    assert c1.name == "User"
    assert c1.parent_type == "BaseEntity"
    assert c1.attributes["name"] == "str"
    assert c1.attributes["age"] == "int"
    assert "name" in c1.required_fields

    # Directive format 2: Relation
    r1 = engine.teach_directive("User -> owns -> Repository")
    assert isinstance(r1, OntologyRelation)
    assert r1.source_entity == "User"
    assert r1.predicate == "owns"
    assert r1.target_entity == "Repository"

    # Directive format 3: Axiom
    ax1 = engine.teach_directive("Rule RequireAdmin: User role == admin")
    assert isinstance(ax1, OntologyAxiom)
    assert ax1.name == "RequireAdmin"
    assert ax1.subject_entity == "User"
    assert ax1.predicate == "role"
    assert ax1.object_value == "admin"

    # Directive format 4: Fail-fast on unparseable directive
    with pytest.raises(UnparseableDirectiveError):
        engine.teach_directive("General descriptive statement about architecture")

    # Directive format 5: Opt-in fallback
    c_fallback = engine.teach_directive(
        "General descriptive statement about architecture", allow_unparsed_concept=True
    )
    assert isinstance(c_fallback, OntologyConcept)


def test_tier_1_deterministic_validation() -> None:
    engine = OntologyEngine()
    engine.teach_concept(
        name="UserProfile",
        attributes={
            "username": "str",
            "age": "int",
            "is_admin": "bool",
            "tags": "list",
            "meta": "dict",
        },
        required_fields=["username", "age"],
    )
    engine.teach_axiom(
        name="AdminCheck",
        subject_entity="UserProfile",
        predicate="is_admin",
        object_value="True",
    )

    pinned_hash = engine.content_hash

    # 1. Valid data
    res_valid = engine.validate_entity(
        "UserProfile",
        {
            "username": "alice",
            "age": 30,
            "is_admin": True,
            "tags": ["dev", "lead"],
            "meta": {"verified": True},
        },
        pinned_content_hash=pinned_hash,
    )
    assert res_valid.is_valid is True
    assert len(res_valid.errors) == 0
    assert res_valid.matched_tier == OntologyTier.ASSERTED
    assert res_valid.latency_ms >= 0.0

    # 2. Missing required field
    res_missing = engine.validate_entity(
        "UserProfile",
        {"username": "alice"},
        pinned_content_hash=pinned_hash,
    )
    assert res_missing.is_valid is False
    assert any("age" in err for err in res_missing.errors)

    # 3. Type mismatch
    res_bad_type = engine.validate_entity(
        "UserProfile",
        {"username": "alice", "age": "thirty", "is_admin": True},
        pinned_content_hash=pinned_hash,
    )
    assert res_bad_type.is_valid is False
    assert any("int" in err for err in res_bad_type.errors)

    # 4. Axiom violation
    res_axiom_fail = engine.validate_entity(
        "UserProfile",
        {"username": "alice", "age": 30, "is_admin": False},
        pinned_content_hash=pinned_hash,
    )
    assert res_axiom_fail.is_valid is False
    assert any("AdminCheck" in err for err in res_axiom_fail.errors)

    # 5. Pinned hash mismatch
    res_hash_fail = engine.validate_entity(
        "UserProfile",
        {"username": "alice", "age": 30},
        pinned_content_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )
    assert res_hash_fail.is_valid is False
    assert any("mismatch" in err for err in res_hash_fail.errors)

    # 6. Unknown entity
    res_unknown = engine.validate_entity("UnknownEntity", {"some": "data"})
    assert res_unknown.is_valid is False
    assert any("not defined" in err for err in res_unknown.errors)


def test_candidate_tier_is_advisory_never_enforcing() -> None:
    engine = OntologyEngine()
    engine.induce_concept(
        name="CandidateEntity",
        attributes={"code": "int"},
        required_fields=["code"],
    )

    # Validating against candidate entity must NOT reject an action
    res = engine.validate_entity("CandidateEntity", {"code": "not_an_int"})
    assert res.is_valid is True  # Non-enforcing!
    assert res.matched_tier == OntologyTier.INDUCED_CANDIDATE
    assert len(res.warnings) > 0


def test_induction_and_contradiction_handling() -> None:
    engine = OntologyEngine()
    engine.teach_concept(
        name="ServiceConfig",
        attributes={"port": "int", "env": "str"},
        parent_type="BaseConfig",
    )

    # Inducing compatible concept
    c1 = engine.induce_concept(
        name="ServiceConfig",
        attributes={"port": "int"},
        source_session="sess_1",
    )
    assert c1.tier == OntologyTier.ASSERTED  # Asserted remains

    # Inducing contradicting concept against ASSERTED -> Raises OntologyContradictionError
    with pytest.raises(OntologyContradictionError, match="contradicts asserted"):
        engine.induce_concept(
            name="ServiceConfig",
            attributes={"port": "str"},  # Conflict with int
            source_session="sess_2",
        )

    # Inducing new candidate
    c_cand = engine.induce_concept(
        name="NewInducedService",
        attributes={"timeout_ms": "int"},
        source_session="sess_1",
    )
    assert c_cand.tier == OntologyTier.INDUCED_CANDIDATE
    assert c_cand.evidence is not None and c_cand.evidence.observation_count == 1

    # Induce again from different session
    c_cand2 = engine.induce_concept(
        name="NewInducedService",
        attributes={"timeout_ms": "int"},
        source_session="sess_2",
    )
    assert c_cand2.evidence is not None and c_cand2.evidence.observation_count == 2

    # Promote to INDUCED_ENFORCING
    promoted = engine.promote("NewInducedService", force=True)
    assert promoted.tier == OntologyTier.INDUCED_ENFORCING

    # Now induce contradiction against INDUCED_ENFORCING -> Immediate Demotion!
    demoted = engine.induce_concept(
        name="NewInducedService",
        attributes={"timeout_ms": "str"},  # Conflict
        source_session="sess_3",
    )
    assert demoted.tier == OntologyTier.INDUCED_CANDIDATE
    assert demoted.evidence is not None
    assert len(demoted.evidence.contradicting_observations) > 0


def test_promotion_criteria_and_demotion() -> None:
    engine = OntologyEngine()
    engine.induce_concept(name="TaskWorker", attributes={"queue": "str"}, source_session="s1")

    # Promotion fails when obs < 5 and not forced
    with pytest.raises(OntologyPromotionError, match="observation count"):
        engine.promote("TaskWorker", min_observations=5)

    # Increment observations across 5 sessions
    for i in range(2, 6):
        engine.induce_concept(name="TaskWorker", source_session=f"s{i}")

    # Now promotion succeeds
    promoted = engine.promote("TaskWorker", min_observations=5)
    assert promoted.tier == OntologyTier.INDUCED_ENFORCING

    # Promote already enforcing
    same_promoted = engine.promote("TaskWorker")
    assert same_promoted.tier == OntologyTier.INDUCED_ENFORCING

    # Demotion
    demoted = engine.demote("TaskWorker", reason="Reviewer found edge cases")
    assert demoted.tier == OntologyTier.INDUCED_CANDIDATE

    # Demoting asserted concept raises error
    engine.teach_concept(name="HumanAsserted")
    with pytest.raises(OntologyViolationError, match="Cannot demote asserted"):
        engine.demote("HumanAsserted")


def test_retraction_forget_and_dependency_blocking() -> None:
    engine = OntologyEngine()
    engine.teach_concept(name="ParentEntity")
    engine.teach_concept(name="ChildEntity", parent_type="ParentEntity")
    engine.teach_relation(
        source_entity="ParentEntity", predicate="links", target_entity="OtherEntity"
    )
    engine.teach_axiom(
        name="ParentAxiom", subject_entity="ParentEntity", predicate="active", object_value="True"
    )

    # Retraction blocked because asserted dependents exist
    with pytest.raises(OntologyRetractionBlockedError, match="blocking asserted dependents"):
        engine.forget("ParentEntity", force=False)

    # Forced retraction cascades
    retracted = engine.forget("ParentEntity", force=True)
    assert "ParentEntity" in retracted
    assert engine.get_concept("ParentEntity") is None
    assert engine.get_concept("ChildEntity") is None


def test_linkml_export_and_yaml_roundtrip() -> None:
    engine = OntologyEngine(
        agent_id="agent_alpha", namespace_iri="https://uclone-x.ai/ontology/alpha"
    )
    engine.teach_concept(
        name="Deployment",
        description="Kubernetes deployment",
        attributes={"replicas": "int", "image": "str"},
        required_fields=["replicas"],
    )
    engine.teach_relation(source_entity="Deployment", predicate="targets", target_entity="Cluster")
    engine.teach_axiom(
        name="MinReplicas", subject_entity="Deployment", predicate="replicas", object_value="2"
    )

    # LinkML export
    linkml_yaml = engine.export_linkml_yaml()
    assert "Deployment" in linkml_yaml
    assert "replicas" in linkml_yaml
    assert "linkml:" in linkml_yaml

    # File persistence
    with TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "agent_alpha.yaml"
        engine.save_to_yaml(save_path)
        assert save_path.is_file()

        # Load back
        engine2 = OntologyEngine(agent_id="agent_alpha")
        engine2.load_from_yaml(save_path)
        assert engine2.version == engine.version
        assert engine2.content_hash == engine.content_hash
        assert engine2.get_concept("Deployment") is not None
        assert len(engine2.list_relations()) == 1
        assert len(engine2.list_axioms()) == 1


@pytest.mark.asyncio
async def test_ontology_inducer() -> None:
    engine = OntologyEngine()
    inducer = OntologyInducer(engine=engine)

    turn_text = "The AuthService authenticated JWTPayload and sent response to ClientApp."
    candidates = await inducer.induce_from_turn(
        turn_text=turn_text,
        tool_results=(),
    )
    assert len(candidates) >= 2
    names = [c.name for c in candidates]
    assert "AuthService" in names or "JWTPayload" in names
    assert all(c.tier == OntologyTier.INDUCED_CANDIDATE for c in candidates)


def test_ontology_tier_normalization_and_unrecognised_rejection() -> None:
    """Ontology models normalize hyphenated spec tiers and reject unrecognised tiers (P6 / Issue #49)."""
    # 1. Hyphenated spec aliases normalize correctly
    c_candidate = OntologyConcept.model_validate({"name": "ConceptA", "tier": "induced-candidate"})
    assert c_candidate.tier == OntologyTier.INDUCED_CANDIDATE
    assert c_candidate.precedence == 10

    c_enforcing = OntologyConcept.model_validate({"name": "ConceptB", "tier": "induced-enforcing"})
    assert c_enforcing.tier == OntologyTier.INDUCED_ENFORCING
    assert c_enforcing.precedence == 50

    r_candidate = OntologyRelation.model_validate(
        {
            "source_entity": "A",
            "predicate": "rel",
            "target_entity": "B",
            "tier": "induced-candidate",
        }
    )
    assert r_candidate.tier == OntologyTier.INDUCED_CANDIDATE

    a_candidate = OntologyAxiom.model_validate(
        {"name": "Ax1", "subject_entity": "A", "tier": "induced-candidate"}
    )
    assert a_candidate.tier == OntologyTier.INDUCED_CANDIDATE

    # 2. Unrecognised tier strings raise ValidationError and are NEVER coerced to ASSERTED
    with pytest.raises(ValidationError, match="Unrecognised ontology tier"):
        OntologyConcept.model_validate({"name": "ConceptC", "tier": "arbitrary-unknown-tier"})

    with pytest.raises(ValidationError, match="Unrecognised ontology tier"):
        OntologyRelation.model_validate(
            {"source_entity": "A", "predicate": "rel", "target_entity": "B", "tier": "fake_tier"}
        )

    with pytest.raises(ValidationError, match="Unrecognised ontology tier"):
        OntologyAxiom.model_validate({"name": "Ax2", "subject_entity": "A", "tier": "invalid_tier"})


def test_ontology_model_copy_immutability_and_hash_recomputation() -> None:
    """model_copy on frozen ontology models preserves immutability, recomputes content_hash, and rejects invalid writes (Issue #42)."""
    engine = OntologyEngine()

    # 1. Initial induction of concept
    c1 = engine.induce_concept(
        name="UserProfile",
        description="Profile entity",
        attributes={"email": "str"},
        required_fields=["email"],
        source_session="sess_1",
    )
    assert isinstance(c1.attributes, MappingProxyType)
    with pytest.raises(TypeError):
        c1.attributes["email"] = "int"  # type: ignore[index]

    initial_hash = compute_concept_hash(
        name="UserProfile",
        parent_type=None,
        attributes={"email": "str"},
        required_fields=["email"],
    )
    assert c1.content_hash == initial_hash

    # 2. Induce and merge additional attributes on the same concept (engine.py:470)
    c2 = engine.induce_concept(
        name="UserProfile",
        attributes={"age": "int"},
        required_fields=["age"],
        source_session="sess_2",
    )
    stored = engine.get_concept("UserProfile")
    assert stored is not None

    # a) Stored concept's attributes is immutable (MappingProxyType / ImmutableStrMapping)
    #    and raises TypeError on attempted mutation after induction / merge.
    assert isinstance(c2.attributes, MappingProxyType)
    assert isinstance(stored.attributes, MappingProxyType)
    assert c2.attributes == {"email": "str", "age": "int"}
    with pytest.raises(TypeError):
        c2.attributes["age"] = "float"  # type: ignore[index]
    with pytest.raises(TypeError):
        stored.attributes["new_field"] = "str"  # type: ignore[index]

    # b) content_hash accurately reflects post-merge attributes (pinned against post-merge hash).
    post_merge_hash = compute_concept_hash(
        name="UserProfile",
        parent_type=None,
        attributes={"email": "str", "age": "int"},
        required_fields=["age", "email"],
    )
    assert c2.content_hash == post_merge_hash
    assert stored.content_hash == post_merge_hash
    assert c2.content_hash != initial_hash

    # c) An invalid derived write via model_copy(update={"name": 12345, "bogus": "yes"}) is rejected with ValidationError.
    with pytest.raises(ValidationError):
        c2.model_copy(update={"name": 12345, "bogus": "yes"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        c2.model_copy(update={"bogus": "yes"})
    with pytest.raises(ValidationError):
        c2.model_copy(update={"name": 12345})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        c2.model_copy(update={"attributes": 12345})  # type: ignore[dict-item]

    # Test model_copy with valid update directly on OntologyConcept
    c_updated = c2.model_copy(update={"description": "Updated profile description"})
    assert c_updated.description == "Updated profile description"
    assert c_updated.content_hash == post_merge_hash
    assert isinstance(c_updated.attributes, MappingProxyType)

    # Test model_copy and hash recomputation on OntologyRelation
    rel = OntologyRelation(
        source_entity="UserProfile",
        predicate="hasRole",
        target_entity="Role",
    )
    assert rel.content_hash == compute_relation_hash("UserProfile", "hasRole", "Role", True)
    rel_updated = rel.model_copy(update={"predicate": "memberOf"})
    assert rel_updated.predicate == "memberOf"
    assert rel_updated.content_hash == compute_relation_hash(
        "UserProfile", "memberOf", "Role", True
    )
    with pytest.raises(ValidationError):
        rel.model_copy(update={"predicate": 12345, "extra": "forbidden"})  # type: ignore[dict-item]

    # Test model_copy and hash recomputation on OntologyAxiom
    axiom = OntologyAxiom(
        name="RuleAge",
        subject_entity="UserProfile",
        predicate="age",
        object_value="18",
    )
    assert axiom.content_hash == compute_axiom_hash("RuleAge", "UserProfile", "age", "18", "")
    ax_updated = axiom.model_copy(update={"object_value": "21"})
    assert ax_updated.object_value == "21"
    assert ax_updated.content_hash == compute_axiom_hash("RuleAge", "UserProfile", "age", "21", "")
    with pytest.raises(ValidationError):
        axiom.model_copy(update={"name": 99999, "unknown_prop": True})  # type: ignore[dict-item]

    # Test model_copy on EvidenceRecord
    evidence = EvidenceRecord(observation_count=2, source_session="s1")
    ev_updated = evidence.model_copy(update={"observation_count": 3})
    assert ev_updated.observation_count == 3
    with pytest.raises(ValidationError):
        evidence.model_copy(update={"observation_count": "not_an_int", "extra": 1})  # type: ignore[dict-item]

    # Test model_copy without update (default passthrough)
    assert c2.model_copy().name == c2.name
    assert rel.model_copy().predicate == rel.predicate
    assert axiom.model_copy().name == axiom.name
    assert evidence.model_copy().observation_count == evidence.observation_count


def test_unrecognised_tier_raises_validation_error() -> None:
    """Issue #49: Unrecognised tier MUST raise ValidationError/ValueError and NEVER coerce to ASSERTED (P6 fail-fast)."""
    with pytest.raises(ValidationError):
        OntologyConcept(name="InvalidConcept", tier="arbitrary_tier")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        OntologyConcept.model_validate({"name": "InvalidConcept", "tier": "arbitrary_tier"})

    with pytest.raises(ValidationError):
        OntologyConcept(name="UnknownConcept", tier="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        OntologyRelation(
            source_entity="A",
            predicate="calls",
            target_entity="B",
            tier="bogus_tier",  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        OntologyRelation.model_validate(
            {
                "source_entity": "A",
                "predicate": "calls",
                "target_entity": "B",
                "tier": "bogus_tier",
            }
        )

    with pytest.raises(ValidationError):
        OntologyAxiom(name="ax1", subject_entity="A", tier="invalid_tier")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        OntologyAxiom.model_validate(
            {
                "name": "ax1",
                "subject_entity": "A",
                "tier": "invalid_tier",
            }
        )


def test_hyphenated_tier_parsing_in_models() -> None:
    """Issue #49: Hyphenated aliases (induced-candidate, induced-enforcing) parse correctly with proper tier and precedence."""
    c = OntologyConcept(name="CandidateConcept", tier="induced-candidate")  # type: ignore[arg-type]
    assert c.tier == OntologyTier.INDUCED_CANDIDATE
    assert c.precedence == 10

    r = OntologyRelation(
        source_entity="ServiceA",
        predicate="calls",
        target_entity="ServiceB",
        tier="induced-candidate",  # type: ignore[arg-type]
    )
    assert r.tier == OntologyTier.INDUCED_CANDIDATE
    assert r.precedence == 10

    a = OntologyAxiom(
        name="CandidateRule",
        subject_entity="ServiceA",
        tier="induced-candidate",  # type: ignore[arg-type]
    )
    assert a.tier == OntologyTier.INDUCED_CANDIDATE
    assert a.precedence == 10

    c_enf = OntologyConcept(name="EnforcingConcept", tier="induced-enforcing")  # type: ignore[arg-type]
    assert c_enf.tier == OntologyTier.INDUCED_ENFORCING
    assert c_enf.precedence == 50

    r_enf = OntologyRelation(
        source_entity="ServiceA",
        predicate="depends_on",
        target_entity="ServiceB",
        tier="induced-enforcing",  # type: ignore[arg-type]
    )
    assert r_enf.tier == OntologyTier.INDUCED_ENFORCING
    assert r_enf.precedence == 50

    a_enf = OntologyAxiom(
        name="EnforcingRule",
        subject_entity="ServiceA",
        tier="induced-enforcing",  # type: ignore[arg-type]
    )
    assert a_enf.tier == OntologyTier.INDUCED_ENFORCING
    assert a_enf.precedence == 50


def test_ontology_service_load_from_yaml_tier_handling() -> None:
    """Issue #49: load_from_yaml parses hyphenated aliases and raises ValidationError on unrecognised tiers."""
    import yaml

    with TemporaryDirectory() as tmp_dir:
        # 1. Hyphenated aliases load correctly
        valid_yaml = Path(tmp_dir) / "valid_hyphen.yaml"
        valid_data = {
            "agent_id": "test_agent",
            "version": 1,
            "concepts": [
                {
                    "name": "HyphenCandidate",
                    "tier": "induced-candidate",
                    "attributes": {"attr": "str"},
                },
                {
                    "name": "HyphenEnforcing",
                    "tier": "induced-enforcing",
                },
            ],
            "relations": [
                {
                    "source_entity": "HyphenCandidate",
                    "predicate": "linksTo",
                    "target_entity": "HyphenEnforcing",
                    "tier": "induced-candidate",
                }
            ],
            "axioms": [
                {
                    "name": "HyphenRule",
                    "subject_entity": "HyphenCandidate",
                    "tier": "induced-enforcing",
                }
            ],
        }
        valid_yaml.write_text(yaml.dump(valid_data), encoding="utf-8")

        service = OntologyService(agent_id="test_agent")
        service.load_from_yaml(valid_yaml)

        c_cand = service.get_concept("HyphenCandidate")
        assert c_cand is not None
        assert c_cand.tier == OntologyTier.INDUCED_CANDIDATE
        assert c_cand.precedence == 10

        c_enf = service.get_concept("HyphenEnforcing")
        assert c_enf is not None
        assert c_enf.tier == OntologyTier.INDUCED_ENFORCING
        assert c_enf.precedence == 50

        assert len(service.list_relations(OntologyTier.INDUCED_CANDIDATE)) == 1
        assert len(service.list_axioms(OntologyTier.INDUCED_ENFORCING)) == 1

        # 2. Unrecognised tier in concept raises ValidationError
        invalid_concept_yaml = Path(tmp_dir) / "invalid_concept.yaml"
        invalid_concept_yaml.write_text(
            yaml.dump(
                {
                    "agent_id": "test_agent",
                    "concepts": [{"name": "BadConcept", "tier": "arbitrary_tier"}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            service.load_from_yaml(invalid_concept_yaml)

        # 3. Unrecognised tier in relation raises ValidationError
        invalid_relation_yaml = Path(tmp_dir) / "invalid_relation.yaml"
        invalid_relation_yaml.write_text(
            yaml.dump(
                {
                    "agent_id": "test_agent",
                    "relations": [
                        {
                            "source_entity": "A",
                            "predicate": "calls",
                            "target_entity": "B",
                            "tier": "bogus_tier",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            service.load_from_yaml(invalid_relation_yaml)

        # 4. Unrecognised tier in axiom raises ValidationError
        invalid_axiom_yaml = Path(tmp_dir) / "invalid_axiom.yaml"
        invalid_axiom_yaml.write_text(
            yaml.dump(
                {
                    "agent_id": "test_agent",
                    "axioms": [
                        {
                            "name": "BadRule",
                            "subject_entity": "A",
                            "tier": "unknown_tier",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            service.load_from_yaml(invalid_axiom_yaml)


def test_ontology_engine_export_graph() -> None:
    """Verify export_graph returns accurate JSON-serializable structure with concepts, relations, axioms."""
    engine = OntologyEngine(agent_id="agent_alpha")
    # Empty export
    empty_graph = engine.export_graph()
    assert empty_graph["concepts"] == []
    assert empty_graph["relations"] == []
    assert empty_graph["axioms"] == []
    assert empty_graph["total_concepts"] == 0
    assert empty_graph["total_axioms"] == 0
    assert empty_graph["total_relations"] == 0
    assert empty_graph["summary"]["total_concepts"] == 0

    # Populate engine
    engine.register_entity(
        OntologyConcept(
            name="AlphaConcept",
            tier=OntologyTier.ASSERTED,
            attributes={"id": "string"},
            required_fields=("id",),
        )
    )
    engine.register_entity(
        OntologyConcept(
            name="BetaConcept",
            tier=OntologyTier.INDUCED_CANDIDATE,
            attributes={"val": "integer"},
            required_fields=("val",),
        )
    )
    engine.register_relation(
        OntologyRelation(
            source_entity="AlphaConcept",
            predicate="links_to",
            target_entity="BetaConcept",
            is_directed=True,
            tier=OntologyTier.ASSERTED,
        )
    )
    engine.register_axiom(
        OntologyAxiom(
            name="RuleAlpha",
            subject_entity="AlphaConcept",
            predicate="id != ''",
            rule_expression="len(id) > 0",
        )
    )

    graph = engine.export_graph()
    assert graph["total_concepts"] == 2
    assert graph["total_relations"] == 1
    assert graph["total_axioms"] == 1
    assert graph["summary"]["asserted_count"] == 1
    assert graph["summary"]["induced_candidate_count"] == 1
    assert len(graph["concepts"]) == 2
    assert len(graph["relations"]) == 1
    assert len(graph["axioms"]) == 1
    assert graph["relations"][0]["source"] == "AlphaConcept"
    assert graph["relations"][0]["target"] == "BetaConcept"
    assert graph["axioms"][0]["name"] == "RuleAlpha"
