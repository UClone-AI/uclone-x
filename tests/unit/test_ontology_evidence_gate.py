"""Unit tests for OntologyEngine promotion evidence gate across axioms, relations, and concepts (Issue #142, P7, P8)."""

from __future__ import annotations

import pytest

from uclone_x.errors import (
    OntologyPromotionError,
    OntologyViolationError,
    PromotionCriteriaNotMetError,
)
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import EvidenceRecord, OntologyTier


def test_evidence_record_session_properties() -> None:
    """EvidenceRecord correctly calculates session_count and distinct_sessions."""
    ev = EvidenceRecord(
        observation_count=5,
        source_session="s1",
        originating_sessions=("s1", "s2", "s1", "s3"),
    )
    assert ev.session_count == 3
    assert ev.distinct_sessions == ("s1", "s2", "s3")

    ev_single = EvidenceRecord(observation_count=3, source_session="s_only")
    assert ev_single.session_count == 1
    assert ev_single.distinct_sessions == ("s_only",)

    ev_empty = EvidenceRecord(observation_count=1)
    assert ev_empty.session_count == 0
    assert ev_empty.distinct_sessions == ()


def test_promote_axiom_fails_when_observation_count_below_threshold() -> None:
    """Promoting an induced axiom with observation_count < min_observations fails (Issue #142)."""
    engine = OntologyEngine(agent_id="test_agent")

    # Induce candidate axiom with 1 observation
    engine.induce_axiom(
        name="RequireTls",
        subject_entity="NetworkService",
        predicate="tls_enabled",
        object_value="true",
        rule_expression="tls_enabled == true",
        source_session="session_1",
    )

    # Attempt promotion without force: fails evidence gate
    with pytest.raises(PromotionCriteriaNotMetError, match="observation count"):
        engine.promote("RequireTls", min_observations=5)

    # Also verify it is an instance of OntologyPromotionError
    with pytest.raises(OntologyPromotionError):
        engine.promote("RequireTls", min_observations=5)

    ax = engine.get_axiom("RequireTls")
    assert ax is not None
    assert ax.tier == OntologyTier.INDUCED_CANDIDATE


def test_promote_axiom_fails_distinct_session_floor() -> None:
    """Promoting an induced axiom with multiple observations but single session fails distinct session floor."""
    engine = OntologyEngine(agent_id="test_agent")

    # Induce candidate axiom 5 times within a single session
    for _ in range(5):
        engine.induce_axiom(
            name="RequireMfa",
            subject_entity="AuthService",
            predicate="mfa_active",
            object_value="true",
            source_session="session_isolated",
        )

    ax = engine.get_axiom("RequireMfa")
    assert ax is not None
    assert ax.evidence is not None
    assert ax.evidence.observation_count == 5
    assert ax.evidence.session_count == 1

    # Attempt promotion: observation count (5 >= 5) passes, but distinct session floor (1 < 2) fails
    with pytest.raises(PromotionCriteriaNotMetError, match="distinct session count"):
        engine.promote("RequireMfa", min_observations=5, min_distinct_sessions=2)

    # Test with min_sessions parameter alias
    with pytest.raises(PromotionCriteriaNotMetError, match="distinct session count"):
        engine.promote("RequireMfa", min_observations=5, min_sessions=3)

    assert ax.tier == OntologyTier.INDUCED_CANDIDATE


def test_promote_axiom_succeeds_with_sufficient_observations_and_sessions() -> None:
    """Promoting an induced axiom with sufficient observations across distinct sessions succeeds."""
    engine = OntologyEngine(agent_id="test_agent")
    initial_version = engine.version

    # Induce candidate axiom across 5 distinct sessions
    for i in range(1, 6):
        engine.induce_axiom(
            name="PositiveBalance",
            subject_entity="Account",
            predicate="balance",
            rule_expression="balance >= 0",
            source_session=f"session_{i}",
        )

    ax_cand = engine.get_axiom("PositiveBalance")
    assert ax_cand is not None
    assert ax_cand.evidence is not None
    assert ax_cand.evidence.observation_count == 5
    assert ax_cand.evidence.session_count == 5

    # Promote to INDUCED_ENFORCING
    promoted = engine.promote("PositiveBalance", min_observations=5, min_distinct_sessions=2)
    assert promoted.tier == OntologyTier.INDUCED_ENFORCING
    assert engine.version == initial_version + 1

    # Promoting already enforcing element returns it idempotently
    same_promoted = engine.promote("PositiveBalance")
    assert same_promoted.tier == OntologyTier.INDUCED_ENFORCING

    # Promote to ASSERTED tier with sufficient evidence
    promoted_asserted = engine.promote(
        "PositiveBalance",
        target_tier=OntologyTier.ASSERTED,
        min_observations=5,
        min_distinct_sessions=2,
    )
    assert promoted_asserted.tier == OntologyTier.ASSERTED


def test_promote_axiom_fails_with_open_contradictions() -> None:
    """Promoting an induced axiom with open contradictions fails promotion."""
    engine = OntologyEngine(agent_id="test_agent")

    # Induce axiom and promote with force
    engine.induce_axiom(
        name="CacheTtlLimit",
        subject_entity="CacheService",
        predicate="ttl",
        object_value="3600",
        source_session="s1",
    )
    engine.promote("CacheTtlLimit", force=True)

    # Induce conflicting value -> triggers automatic demotion to INDUCED_CANDIDATE
    demoted = engine.induce_axiom(
        name="CacheTtlLimit",
        subject_entity="CacheService",
        predicate="ttl",
        object_value="7200",  # Conflicting value
        source_session="s2",
    )
    assert demoted.tier == OntologyTier.INDUCED_CANDIDATE
    assert demoted.evidence is not None
    assert len(demoted.evidence.contradicting_observations) > 0

    # Induce more observations in other sessions
    for i in range(3, 8):
        engine.induce_axiom(
            name="CacheTtlLimit",
            subject_entity="CacheService",
            source_session=f"s{i}",
        )

    # Promotion fails because open contradictions exist
    with pytest.raises(PromotionCriteriaNotMetError, match="open contradictions present"):
        engine.promote("CacheTtlLimit", min_observations=5, min_distinct_sessions=2)


def test_promote_relation_evidence_gate() -> None:
    """Promoting relations enforces the exact same evidence gate (Issue #142)."""
    engine = OntologyEngine(agent_id="test_agent")

    # 1. Induce relation with 1 observation
    engine.induce_relation(
        source_entity="User",
        predicate="manages",
        target_entity="Project",
        source_session="session_1",
    )

    rel_name = "User:manages:Project"

    # Fails observation count check
    with pytest.raises(PromotionCriteriaNotMetError, match="observation count"):
        engine.promote(rel_name, min_observations=5)

    # 2. Induce 4 more times in the SAME session
    for _ in range(4):
        engine.induce_relation(
            source_entity="User",
            predicate="manages",
            target_entity="Project",
            source_session="session_1",
        )

    # Fails distinct session floor check (obs=5, sess=1 < 2)
    with pytest.raises(PromotionCriteriaNotMetError, match="distinct session count"):
        engine.promote(rel_name, min_observations=5, min_distinct_sessions=2)

    # 3. Induce across distinct sessions
    for i in range(2, 6):
        engine.induce_relation(
            source_entity="User",
            predicate="manages",
            target_entity="Project",
            source_session=f"session_{i}",
        )

    # Promotion succeeds! Test with arrow format identifier
    rel_arrow = "User -> manages -> Project"
    promoted = engine.promote(rel_arrow, min_observations=5, min_distinct_sessions=2)
    assert promoted.tier == OntologyTier.INDUCED_ENFORCING


def test_force_promotion_bypasses_evidence_gate() -> None:
    """force=True cleanly overrides observation and session thresholds for concepts, axioms, and relations."""
    engine = OntologyEngine(agent_id="test_agent")

    # Concept with 1 observation
    engine.induce_concept(name="UnconfirmedEntity", source_session="s1")
    promoted_c = engine.promote("UnconfirmedEntity", force=True)
    assert promoted_c.tier == OntologyTier.INDUCED_ENFORCING

    # Axiom with 1 observation
    engine.induce_axiom(
        name="UnconfirmedRule",
        subject_entity="UnconfirmedEntity",
        predicate="flag",
        object_value="1",
        source_session="s1",
    )
    promoted_a = engine.promote("UnconfirmedRule", force=True)
    assert promoted_a.tier == OntologyTier.INDUCED_ENFORCING

    # Relation with 1 observation
    engine.induce_relation(
        source_entity="UnconfirmedEntity",
        predicate="depends_on",
        target_entity="OtherEntity",
        source_session="s1",
    )
    promoted_r = engine.promote("UnconfirmedEntity:depends_on:OtherEntity", force=True)
    assert promoted_r.tier == OntologyTier.INDUCED_ENFORCING


def test_demote_axiom_and_relation_updates_evidence() -> None:
    """Demoting enforcing axioms and relations updates evidence with contradiction reasons."""
    engine = OntologyEngine(agent_id="test_agent")

    engine.induce_axiom(name="RuleA", subject_entity="Service")
    engine.promote("RuleA", force=True)

    demoted_a = engine.demote("RuleA", reason="Security review flag")
    assert demoted_a.tier == OntologyTier.INDUCED_CANDIDATE
    assert demoted_a.evidence is not None
    assert "Security review flag" in demoted_a.evidence.contradicting_observations

    engine.induce_relation(source_entity="S1", predicate="links", target_entity="S2")
    engine.promote("S1:links:S2", force=True)

    demoted_r = engine.demote("S1:links:S2", reason="Architecture change")
    assert demoted_r.tier == OntologyTier.INDUCED_CANDIDATE
    assert demoted_r.evidence is not None
    assert "Architecture change" in demoted_r.evidence.contradicting_observations


def test_promote_and_demote_asserted_errors() -> None:
    """Attempting to promote or demote asserted elements raises appropriate errors."""
    engine = OntologyEngine(agent_id="test_agent")

    engine.teach_concept(name="AssertedConcept")
    engine.teach_axiom(name="AssertedAxiom", subject_entity="AssertedConcept")
    engine.teach_relation(source_entity="AssertedConcept", predicate="has", target_entity="Other")

    # Promoting already asserted element raises OntologyPromotionError
    with pytest.raises(OntologyPromotionError, match="already asserted"):
        engine.promote("AssertedConcept")

    with pytest.raises(OntologyPromotionError, match="already asserted"):
        engine.promote("AssertedAxiom")

    with pytest.raises(OntologyPromotionError, match="already asserted"):
        engine.promote("AssertedConcept:has:Other")

    # Demoting asserted element raises OntologyViolationError
    with pytest.raises(OntologyViolationError, match="Cannot demote asserted"):
        engine.demote("AssertedConcept")

    with pytest.raises(OntologyViolationError, match="Cannot demote asserted"):
        engine.demote("AssertedAxiom")

    with pytest.raises(OntologyViolationError, match="Cannot demote asserted"):
        engine.demote("AssertedConcept:has:Other")


def test_promote_non_existent_element_raises_error() -> None:
    """Promoting a non-existent ontology element raises OntologyPromotionError."""
    engine = OntologyEngine(agent_id="test_agent")

    with pytest.raises(OntologyPromotionError, match="not found for promotion"):
        engine.promote("NonExistentTerm")
