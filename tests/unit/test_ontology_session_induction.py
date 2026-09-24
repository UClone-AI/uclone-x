# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Unit tests for autonomous domain knowledge induction from conversation sessions (Issue #57)."""

from pathlib import Path

from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.ontology.engine import OntologyEngine, SessionKnowledgeExtractor
from uclone_x.ontology.models import OntologyTier


def test_session_knowledge_extraction_from_messages() -> None:
    engine = OntologyEngine(agent_id="test_agent")
    extractor = SessionKnowledgeExtractor(engine=engine)

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Please check CustomerProfile and AccountBalance for User #123",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="CustomerProfile validated with positive AccountBalance.",
        ),
    ]

    candidates = extractor.extract_from_session(
        messages, session_id="sess_101", model_id="mock_model"
    )
    assert len(candidates) >= 2
    concept_names = {c.name for c in candidates}
    assert "CustomerProfile" in concept_names
    assert "AccountBalance" in concept_names

    # Check that candidate concepts are staged in CANDIDATE tier
    for c in candidates:
        assert c.tier == OntologyTier.INDUCED_CANDIDATE
        assert c.evidence is not None
        assert c.evidence.source_session == "sess_101"
        assert c.evidence.model_id == "mock_model"
        assert len(c.content_hash) == 64


def test_session_knowledge_prompt_injection() -> None:
    engine = OntologyEngine(agent_id="test_agent")
    extractor = SessionKnowledgeExtractor(engine=engine)

    # Empty ontology should return original prompt
    base_prompt = "You are a helpful banking assistant."
    assert extractor.inject_rules_to_prompt(base_prompt) == base_prompt

    # Add Asserted concept and Enforcing axiom
    engine.assert_concept(name="Transaction", description="Financial transfer record")
    engine.assert_axiom(
        name="NoNegativeBalance",
        subject_entity="Account",
        predicate="has_balance",
        rule_expression="balance >= 0",
    )

    enriched_prompt = extractor.inject_rules_to_prompt(base_prompt)
    assert "[Active Domain Ontology Invariants]:" in enriched_prompt
    assert "Transaction" in enriched_prompt
    assert "NoNegativeBalance" in enriched_prompt
    assert "balance >= 0" in enriched_prompt


def test_session_knowledge_persistence_to_yaml(tmp_path: Path) -> None:
    engine = OntologyEngine(agent_id="persisted_agent")
    extractor = SessionKnowledgeExtractor(engine=engine)

    messages = [
        ChatMessage(role=MessageRole.USER, content="Create OrderRecord for shipment"),
    ]
    extractor.extract_from_session(messages, session_id="sess_202")

    yaml_file = tmp_path / "ontology.yaml"
    engine.save_to_yaml(yaml_file)
    assert yaml_file.is_file()

    # Load into fresh engine
    new_engine = OntologyEngine()
    new_engine.load_from_yaml(yaml_file)
    concept = new_engine.get_concept("OrderRecord")
    assert concept is not None
    assert concept.tier == OntologyTier.INDUCED_CANDIDATE
    assert concept.evidence is not None
    assert concept.evidence.source_session == "sess_202"
