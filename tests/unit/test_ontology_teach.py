"""Unit tests for ontology directive parsing, invariant rule creation, and P6 fail-fast errors (Issue #144)."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest
from typer.testing import CliRunner

from uclone_x.cli.main import app
from uclone_x.errors import OntologyError, UnparseableDirectiveError
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import (
    OntologyAxiom,
    OntologyConcept,
    OntologyInvariant,
    OntologyRelation,
    OntologyTier,
)

runner = CliRunner()


def test_teach_directive_structured_constraint_grpc_mtls() -> None:
    """Structured constraint directive parses into an OntologyInvariant / OntologyAxiom without junk concepts."""
    engine = OntologyEngine(agent_id="agt_sec")
    result = engine.teach_directive("all internal gRPC calls must use mTLS encryption")

    assert isinstance(result, OntologyInvariant)
    assert isinstance(result, OntologyAxiom)
    assert result.name == "all_internal_grpc_calls_must_use_mtls_encryption"
    assert result.subject_entity == "internal gRPC calls"
    assert result.predicate == "use"
    assert result.object_value == "mTLS encryption"
    assert result.rule_expression == "all internal gRPC calls must use mTLS encryption"
    assert result.tier == OntologyTier.ASSERTED
    assert result.precedence == 100

    # Stored in axioms
    assert engine.get_axiom(result.name) is not None
    # No junk concepts created
    assert len(engine.list_concepts()) == 0


def test_teach_directive_whenever_trigger() -> None:
    """Whenever <X>, <Y> parses into an invariant rule trigger."""
    engine = OntologyEngine()
    result = engine.teach_directive("whenever token expires, refresh token")

    assert isinstance(result, OntologyInvariant)
    assert result.name == "whenever_token_expires_refresh_token"
    assert result.subject_entity == "token expires"
    assert result.predicate == "triggers"
    assert result.object_value == "refresh token"
    assert result.rule_expression == "whenever token expires, then refresh token"
    assert len(engine.list_concepts()) == 0


def test_teach_directive_requires_rule() -> None:
    """<X> requires <Y> parses into an invariant requirement rule."""
    engine = OntologyEngine()
    result = engine.teach_directive("Deployment requires replicas")

    assert isinstance(result, OntologyInvariant)
    assert result.name == "deployment_requires_replicas"
    assert result.subject_entity == "Deployment"
    assert result.predicate == "requires"
    assert result.object_value == "replicas"
    assert len(engine.list_concepts()) == 0


def test_teach_directive_concept_is_a_syntax() -> None:
    """'<X> is a <Y>' parses cleanly into an OntologyConcept with parent_type set."""
    engine = OntologyEngine()
    result = engine.teach_directive("User is a Person")

    assert isinstance(result, OntologyConcept)
    assert result.name == "User"
    assert result.parent_type == "Person"
    assert result.description == "User is a Person"
    assert result.tier == OntologyTier.ASSERTED
    assert engine.get_concept("User") is not None
    assert len(engine.list_axioms()) == 0


def test_teach_directive_concept_formal_syntax() -> None:
    """Formal 'concept <Name> extends <Parent> with attributes ... requires ...' syntax parses."""
    engine = OntologyEngine()
    result = engine.teach_directive(
        "Concept User extends BaseEntity with attributes name:str, age:int requires name"
    )

    assert isinstance(result, OntologyConcept)
    assert result.name == "User"
    assert result.parent_type == "BaseEntity"
    assert result.attributes == {"name": "str", "age": "int"}
    assert result.required_fields == ("name",)
    assert result.tier == OntologyTier.ASSERTED


def test_teach_directive_define_concept_syntax() -> None:
    """'define concept <Name>: <description>' parses into an OntologyConcept."""
    engine = OntologyEngine()
    result = engine.teach_directive("define concept Customer: A paying account holder")

    assert isinstance(result, OntologyConcept)
    assert result.name == "Customer"
    assert result.description == "A paying account holder"
    assert result.tier == OntologyTier.ASSERTED


def test_teach_directive_relation_triplet() -> None:
    """'<Source> -> <predicate> -> <Target>' parses into an OntologyRelation."""
    engine = OntologyEngine()
    result = engine.teach_directive("User -> owns -> Repository")

    assert isinstance(result, OntologyRelation)
    assert result.source_entity == "User"
    assert result.predicate == "owns"
    assert result.target_entity == "Repository"
    assert result.tier == OntologyTier.ASSERTED


def test_teach_directive_explicit_axiom_syntax() -> None:
    """'Rule <Name>: <subject> requires <predicate> == <value>' parses into an OntologyAxiom."""
    engine = OntologyEngine()
    result = engine.teach_directive("Rule RequireAdmin: User role == admin")

    assert isinstance(result, OntologyAxiom)
    assert result.name == "RequireAdmin"
    assert result.subject_entity == "User"
    assert result.predicate == "role"
    assert result.object_value == "admin"


def test_teach_directive_unparseable_fails_fast() -> None:
    """Unparseable arbitrary sentences raise UnparseableDirectiveError without creating junk concepts (P6)."""
    engine = OntologyEngine()

    unparseable_samples = [
        "General descriptive statement about architecture",
        "Hello world this is some random note",
        "The quick brown fox jumps over the lazy dog",
        "12345",
        "",
    ]

    for sample in unparseable_samples:
        with pytest.raises(UnparseableDirectiveError) as exc_info:
            engine.teach_directive(sample)

        err = exc_info.value
        assert isinstance(err, OntologyError)
        assert err.text == sample.strip()
        assert len(err.expected_forms) > 0
        assert "Cannot parse directive" in str(err)

    # Engine knowledge graph remains clean: 0 concepts, 0 relations, 0 axioms
    assert len(engine.list_concepts()) == 0
    assert len(engine.list_relations()) == 0
    assert len(engine.list_axioms()) == 0


def test_teach_directive_allow_unparsed_concept_opt_in() -> None:
    """When allow_unparsed_concept=True is explicitly set, fallback concept creation is allowed."""
    engine = OntologyEngine()
    result = engine.teach_directive(
        "General descriptive statement about architecture",
        allow_unparsed_concept=True,
    )

    assert isinstance(result, OntologyConcept)
    assert result.description == "General descriptive statement about architecture"
    assert len(engine.list_concepts()) == 1


def test_cli_ontology_teach_structured_directive_and_error_handling() -> None:
    """CLI teach command handles structured directives, explicit concept flags, and unparseable errors."""
    with TemporaryDirectory() as tmp_dir:
        # 1. Structured constraint directive
        res_c = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "all internal gRPC calls must use mTLS encryption",
                "--dir",
                tmp_dir,
                "--agent",
                "sec_agent",
            ],
        )
        assert res_c.exit_code == 0
        assert "Taught asserted element" in res_c.stdout
        assert "all_internal_grpc_calls_must_use_mtls_encryption" in res_c.stdout

        # 2. Concept is-a directive
        res_isa = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "User is a Person",
                "--dir",
                tmp_dir,
                "--agent",
                "sec_agent",
            ],
        )
        assert res_isa.exit_code == 0
        assert "Taught asserted element" in res_isa.stdout
        assert "User" in res_isa.stdout

        # 3. Explicit --concept flag
        res_flag = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--concept",
                "PaymentGateway",
                "--dir",
                tmp_dir,
                "--agent",
                "sec_agent",
            ],
        )
        assert res_flag.exit_code == 0
        assert "PaymentGateway" in res_flag.stdout

        # 4. Unparseable arbitrary sentence fails with exit code 1 and red error message
        res_err = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "General descriptive statement about architecture without syntax",
                "--dir",
                tmp_dir,
                "--agent",
                "sec_agent",
            ],
        )
        assert res_err.exit_code == 1
        assert "Unparseable directive" in res_err.stdout
