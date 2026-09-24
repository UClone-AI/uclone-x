"""Unit tests for strict ontology tier filtering and active invariant injection in BaseAgent reasoning (Issue #128, P7, P8)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.ontology.engine import OntologyEngine, SessionKnowledgeExtractor
from uclone_x.ontology.models import (
    OntologyAxiom,
    OntologyInvariant,
    OntologyTier,
)


def _mock_llm_response(content: str = "Mock response") -> tuple[MagicMock, list[LLMRequest]]:
    captured_requests: list[LLMRequest] = []
    mock_llm = MagicMock(spec=LLMProviderProtocol)

    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="mock", model="mock-model"),
        served_by=ServiceRef(provider="mock", model="mock-model"),
    )
    resp = ModelResponse(
        content=content,
        usage=TokenUsage(provider="mock", model="mock-model", input_tokens=10, output_tokens=20),
        finish_reason=FinishReason.STOP,
        model_name="mock-model",
        provenance=prov,
    )

    async def _generate(req: LLMRequest) -> ModelResponse:
        captured_requests.append(req)
        return resp

    mock_llm.generate = AsyncMock(side_effect=_generate)
    return mock_llm, captured_requests


def test_get_active_invariants_tier_filtering() -> None:
    """get_active_invariants returns only asserted/enforcing rules by default and strictly isolates candidates."""
    engine = OntologyEngine(agent_id="test_agent")

    # 1. Asserted rules across various asserted tiers
    ax_core = engine.teach_axiom(
        name="RequireTls",
        subject_entity="NetworkService",
        predicate="tls_enabled",
        object_value="true",
        rule_expression="tls_enabled == true",
        domain="security",
        tier=OntologyTier.ASSERTED_CORE,
    )
    ax_domain = engine.teach_axiom(
        name="RequireMfa",
        subject_entity="UserSession",
        predicate="mfa_verified",
        object_value="true",
        domain="security",
        tier=OntologyTier.ASSERTED_DOMAIN,
    )
    ax_asserted = engine.teach_axiom(
        name="PositiveBalance",
        subject_entity="Account",
        predicate="balance",
        rule_expression="balance >= 0",
        domain="billing",
        tier=OntologyTier.ASSERTED,
    )

    # 2. Candidate rules (induced from turns/sessions)
    ax_cand1 = engine.induce_axiom(
        name="SuggestedRateLimit",
        subject_entity="ApiEndpoint",
        rule_expression="rate_limit <= 100",
        domain="security",
    )
    ax_cand2 = engine.induce_axiom(
        name="SuggestedAutoArchive",
        subject_entity="Order",
        predicate="status",
        object_value="archived",
        domain="operations",
    )

    assert isinstance(ax_core, OntologyInvariant)
    assert isinstance(ax_domain, OntologyInvariant)
    assert isinstance(ax_asserted, OntologyInvariant)
    assert isinstance(ax_cand1, OntologyAxiom)
    assert isinstance(ax_cand2, OntologyAxiom)

    # 3. Default tier_filter='asserted': returns only asserted rules
    active_default = engine.get_active_invariants()
    active_names = {a.name for a in active_default}
    assert active_names == {"RequireTls", "RequireMfa", "PositiveBalance"}
    assert "SuggestedRateLimit" not in active_names
    assert "SuggestedAutoArchive" not in active_names

    # Explicit tier_filter='asserted'
    active_asserted = engine.get_active_invariants(tier_filter="asserted")
    assert {a.name for a in active_asserted} == active_names

    # 4. tier_filter='candidate': returns only candidate rules
    candidates = engine.get_active_invariants(tier_filter="candidate")
    candidate_names = {a.name for a in candidates}
    assert candidate_names == {"SuggestedRateLimit", "SuggestedAutoArchive"}
    assert "RequireTls" not in candidate_names
    assert "PositiveBalance" not in candidate_names

    # 5. tier_filter='all': returns both asserted and candidate rules
    all_rules = engine.get_active_invariants(tier_filter="all")
    all_names = {a.name for a in all_rules}
    assert all_names == {
        "RequireTls",
        "RequireMfa",
        "PositiveBalance",
        "SuggestedRateLimit",
        "SuggestedAutoArchive",
    }

    # 6. Invalid tier_filter raises ValueError (P6 fail-fast)
    with pytest.raises(ValueError, match="Invalid tier_filter"):
        engine.get_active_invariants(tier_filter="unknown_filter")  # type: ignore[arg-type]


def test_get_active_invariants_domain_filtering() -> None:
    """get_active_invariants supports filtering by domain or subject entity."""
    engine = OntologyEngine(agent_id="test_agent")

    engine.teach_axiom(
        name="SecRule1",
        subject_entity="SecurityService",
        rule_expression="auth == true",
        domain="security",
        tier=OntologyTier.ASSERTED,
    )
    engine.teach_axiom(
        name="SecRule2",
        subject_entity="SecurityService",
        rule_expression="encryption == true",
        domain="security",
        tier=OntologyTier.ASSERTED_DOMAIN,
    )
    engine.teach_axiom(
        name="BillingRule",
        subject_entity="BillingService",
        rule_expression="currency == USD",
        domain="billing",
        tier=OntologyTier.ASSERTED,
    )
    engine.induce_axiom(
        name="CandSecRule",
        subject_entity="SecurityService",
        rule_expression="jwt == valid",
        domain="security",
    )

    # Filter security domain with asserted tier
    sec_asserted = engine.get_active_invariants(domain="security", tier_filter="asserted")
    assert {a.name for a in sec_asserted} == {"SecRule1", "SecRule2"}

    # Filter billing domain with asserted tier
    billing_asserted = engine.get_active_invariants(domain="billing", tier_filter="asserted")
    assert {a.name for a in billing_asserted} == {"BillingRule"}

    # Filter security domain with candidate tier
    sec_candidate = engine.get_active_invariants(domain="security", tier_filter="candidate")
    assert {a.name for a in sec_candidate} == {"CandSecRule"}


@pytest.mark.asyncio
async def test_base_agent_reasoning_injects_asserted_and_excludes_candidate_invariants() -> None:
    """BaseAgent reasoning turn prompt context contains asserted invariants and strictly excludes candidates (P7, P8)."""
    engine = OntologyEngine(agent_id="agent-reasoning-test")

    # Asserted rule
    engine.teach_axiom(
        name="EnforceMtls",
        subject_entity="Microservice",
        predicate="mtls_auth",
        object_value="required",
        rule_expression="mtls_auth == required",
        tier=OntologyTier.ASSERTED,
    )

    # Candidate rule
    engine.induce_axiom(
        name="UnreviewedSuggestion",
        subject_entity="Microservice",
        predicate="cache_ttl",
        object_value="3600",
        rule_expression="cache_ttl <= 3600",
    )

    mock_llm, captured_reqs = _mock_llm_response("Task completed successfully.")

    config = AgentConfig(
        agent_id="agent-reasoning-test",
        name="Reasoning Agent",
        system_prompt="You are a microservice orchestration agent.",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    agent = BaseAgent(config=config, llm=mock_llm, ontology=engine)
    assert agent.ontology is engine

    result = await agent.execute_turn("Deploy the authentication service")
    assert result.is_completed is True
    assert len(captured_reqs) == 1

    llm_req = captured_reqs[0]
    system_msg = llm_req.messages[0]
    assert system_msg.content is not None

    # Asserted invariant is present in the active reasoning context
    assert "[Active Domain Ontology Invariants]:" in system_msg.content
    assert "EnforceMtls" in system_msg.content
    assert "mtls_auth == required" in system_msg.content

    # Candidate invariant is strictly excluded
    assert "UnreviewedSuggestion" not in system_msg.content
    assert "cache_ttl" not in system_msg.content

    # Base prompt history remains clean and unpolluted
    assert agent._history[0].content == "You are a microservice orchestration agent."  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_candidate_rules_become_active_only_after_formal_promotion() -> None:
    """Candidate rules are excluded from agent reasoning until formally promoted to asserted/enforcing tier."""
    engine = OntologyEngine(agent_id="agent-promo-test")

    # Induce candidate invariant
    engine.induce_axiom(
        name="StrictRetryLimit",
        subject_entity="HttpClient",
        rule_expression="max_retries <= 3",
        source_session="sess_turn_1",
    )

    # 1. Before promotion: candidate is excluded
    assert engine.get_active_invariants(tier_filter="asserted") == []
    assert len(engine.get_active_invariants(tier_filter="candidate")) == 1

    mock_llm, captured_reqs = _mock_llm_response("Awaiting instructions.")
    config = AgentConfig(
        agent_id="agent-promo-test",
        name="Promo Test Agent",
        system_prompt="Core agent instructions.",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=config, llm=mock_llm, ontology=engine)

    # Turn 1: candidate rule is NOT in prompt
    await agent.execute_turn("Check retry policy")
    turn_1_sys_content = captured_reqs[0].messages[0].content or ""
    assert "StrictRetryLimit" not in turn_1_sys_content
    assert "[Active Domain Ontology Invariants]:" not in turn_1_sys_content

    # 2. Formally promote the candidate axiom to ASSERTED tier
    promoted = engine.promote("StrictRetryLimit", force=True, target_tier=OntologyTier.ASSERTED)
    assert promoted.tier == OntologyTier.ASSERTED

    # Now get_active_invariants(tier_filter="asserted") returns the promoted axiom
    active_invariants = engine.get_active_invariants(tier_filter="asserted")
    assert len(active_invariants) == 1
    assert active_invariants[0].name == "StrictRetryLimit"
    assert active_invariants[0].tier == OntologyTier.ASSERTED

    # Turn 2: promoted rule is now ACTIVE and injected into the prompt
    await agent.execute_turn("Execute with retry policy")
    assert len(captured_reqs) == 2
    turn_2_sys_content = captured_reqs[1].messages[0].content or ""
    assert "[Active Domain Ontology Invariants]:" in turn_2_sys_content
    assert "StrictRetryLimit" in turn_2_sys_content
    assert "max_retries <= 3" in turn_2_sys_content


def test_session_knowledge_extractor_tier_filtering() -> None:
    """SessionKnowledgeExtractor.inject_rules_to_prompt respects tier_filter and domain parameters."""
    engine = OntologyEngine(agent_id="extractor_test")
    extractor = SessionKnowledgeExtractor(engine=engine)

    engine.teach_axiom(
        name="AssertedRule",
        subject_entity="Service",
        rule_expression="ready == true",
        domain="core",
    )
    engine.induce_axiom(
        name="CandidateRule",
        subject_entity="Service",
        rule_expression="timeout == 5000",
        domain="core",
    )

    base_prompt = "You are a service worker."

    # Default: asserted only
    prompt_asserted = extractor.inject_rules_to_prompt(base_prompt)
    assert "AssertedRule" in prompt_asserted
    assert "CandidateRule" not in prompt_asserted

    # Candidate tier
    prompt_candidate = extractor.inject_rules_to_prompt(base_prompt, tier_filter="candidate")
    assert "CandidateRule" in prompt_candidate
    assert "AssertedRule" not in prompt_candidate

    # All tiers
    prompt_all = extractor.inject_rules_to_prompt(base_prompt, tier_filter="all")
    assert "AssertedRule" in prompt_all
    assert "CandidateRule" in prompt_all
